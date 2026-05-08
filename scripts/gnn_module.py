"""TaxonomyGNN: encode the (frozen) NCBI canonical-rank lineage DAG.

The forward pass takes the static 768-d initial node features built by
``build_taxonomy_graph.py``, runs ``num_layers`` of GCN or GAT message-passing
over the (parent->child + reverse) edges, and returns the per-node embedding
projected to ``out_dim``.

It also exposes a ``species_emb`` helper that gathers a batch's embeddings
either from the leaf-species node only (``fusion='leaf'``) or by concatenating
the species + genus + family ancestors (``fusion='hier'``) — implementing the
F1 vs F2 fusion strategies discussed in the plan.

Note on caching: the graph has ~700 nodes. Forwarding it on every batch is
cheap (microseconds), so we don't bother with manual caching during training.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, GCNConv


# ---------------------------------------------------------------------------
# The taxonomy GNN itself.
# ---------------------------------------------------------------------------
class TaxonomyGNN(nn.Module):
    """Encode the taxonomy DAG; output ``[N_nodes, out_dim]``.

    Parameters
    ----------
    in_dim:
        Dimensionality of the per-node initial features (PubMedBERT = 768).
    hidden:
        GNN hidden width. The plan recommends 128.
    out_dim:
        Output projection dim used downstream (default 64; matches the
        gnn_instruction default).
    num_layers:
        Number of GNN message-passing layers (>=1). 2 is recommended; 3+ tends
        to oversmooth a small DAG of 700 nodes.
    gnn_type:
        'gcn' (default, isotropic) or 'gat' (heads anisotropic, sees parent
        vs sibling differently).
    heads:
        GAT only — number of attention heads per layer.
    dropout:
        Dropout on hidden GNN activations.
    freeze_init:
        If True (default) the 768-d PubMedBERT vectors are stored as a frozen
        buffer and not updated. The down-projection ``input_proj`` and all GNN
        layers remain trainable.
    """

    def __init__(
        self,
        init_features: torch.Tensor,           # [N, in_dim] -- saved as buffer
        edge_index: torch.Tensor,              # [2, 2E]    -- saved as buffer
        in_dim: int = 768,
        hidden: int = 128,
        out_dim: int = 64,
        num_layers: int = 2,
        gnn_type: Literal["gcn", "gat"] = "gcn",
        heads: int = 4,
        dropout: float = 0.1,
        freeze_init: bool = True,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        assert init_features.dim() == 2 and init_features.size(1) == in_dim, (
            f"init_features shape {tuple(init_features.shape)} does not match in_dim={in_dim}"
        )

        self.in_dim = in_dim
        self.hidden = hidden
        self.out_dim = out_dim
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.freeze_init = freeze_init

        # Static graph data -- registered as buffers so .to(device) moves them
        # automatically and they get checkpointed alongside the model.
        if freeze_init:
            # ``register_buffer`` keeps the tensor on the right device but does
            # NOT register it as a Parameter -- so it's not updated by the
            # optimizer. The downstream input_proj is trainable.
            self.register_buffer("init_features", init_features.float(), persistent=False)
        else:
            # If we want to fine-tune the per-node features (rarely useful with
            # only ~700 supervised gradient signals), expose them as a Parameter.
            self.init_features = nn.Parameter(init_features.float().clone())
        self.register_buffer("edge_index", edge_index.long(), persistent=False)

        # Project frozen PubMedBERT -> GNN hidden width.
        self.input_proj = nn.Linear(in_dim, hidden)

        # GNN layers.
        self.layers = nn.ModuleList()
        if gnn_type == "gcn":
            for _ in range(num_layers):
                self.layers.append(GCNConv(hidden, hidden))
        elif gnn_type == "gat":
            # We force concat=False for every layer so the output stays at
            # ``hidden`` -- this keeps the architecture comparable to GCN.
            for _ in range(num_layers):
                self.layers.append(
                    GATConv(hidden, hidden, heads=heads, concat=False, dropout=dropout)
                )
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}")

        # Final projection used to keep the channel dim modest before fusion.
        self.output_proj = nn.Linear(hidden, out_dim)

    # ------------------------------------------------------------------
    def forward(self) -> torch.Tensor:
        """Return all-node embeddings ``[N, out_dim]``.

        Called once per training step (the graph is small).
        """
        x = self.input_proj(self.init_features)                  # [N, hidden]
        for i, layer in enumerate(self.layers):
            x = layer(x, self.edge_index)                        # [N, hidden]
            if i < len(self.layers) - 1:
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return self.output_proj(x)                               # [N, out_dim]


# ---------------------------------------------------------------------------
# Lightweight wrapper that bundles the GNN + fusion logic, exposing a single
# ``forward(batch_node_ids_or_ancestors) -> [B, fused_dim]``. Built for direct
# use inside ``ESM2.forward`` so the rest of the training loop stays untouched.
# ---------------------------------------------------------------------------
class TaxonomySpeciesEncoder(nn.Module):
    """End-to-end species channel: GNN + (leaf|hier) fusion + final projection.

    fusion = 'leaf'  → species_emb only          (F1 in the plan)
    fusion = 'hier'  → concat(species, genus, family) + Linear  (F2)

    The output dim is always ``out_dim`` so swapping fusion strategies does not
    require touching the downstream MLP head.
    """

    HIER_LEVELS = ("species", "genus", "family")
    LEVEL_TO_IDX = {
        "domain": 0, "kingdom": 1, "phylum": 2, "class": 3,
        "order": 4, "family": 5, "genus": 6, "species": 7,
    }

    def __init__(
        self,
        init_features: torch.Tensor,
        edge_index: torch.Tensor,
        ancestors_per_species: torch.Tensor,    # [S, 8]
        species_to_sp_idx: Dict[str, int],      # name -> row in ancestors_per_species
        in_dim: int = 768,
        hidden: int = 128,
        out_dim: int = 64,
        num_layers: int = 2,
        gnn_type: str = "gcn",
        heads: int = 4,
        dropout: float = 0.1,
        fusion: Literal["leaf", "hier"] = "leaf",
        freeze_init: bool = True,
    ):
        super().__init__()
        self.gnn = TaxonomyGNN(
            init_features=init_features,
            edge_index=edge_index,
            in_dim=in_dim,
            hidden=hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            gnn_type=gnn_type,
            heads=heads,
            dropout=dropout,
            freeze_init=freeze_init,
        )

        self.register_buffer("ancestors_per_species", ancestors_per_species.long(), persistent=False)
        # Stable mapping species name -> row in ancestors_per_species.
        self.species_to_sp_idx: Dict[str, int] = dict(species_to_sp_idx)

        self.fusion = fusion
        self.out_dim = out_dim

        if fusion == "hier":
            # Concat(species, genus, family) -> Linear -> out_dim. Missing
            # ranks (-1 in ancestors row) get a learnable rank-specific
            # "absent" vector so the model learns a sensible default for those
            # rare cases (e.g. NCBI does not assign that rank for a clade).
            self.absent_emb = nn.Parameter(torch.zeros(len(self.HIER_LEVELS), out_dim))
            nn.init.normal_(self.absent_emb, std=0.02)
            self.hier_proj = nn.Linear(out_dim * len(self.HIER_LEVELS), out_dim)
            self.hier_levels_idx = [self.LEVEL_TO_IDX[l] for l in self.HIER_LEVELS]
        else:
            self.absent_emb = None
            self.hier_proj = None
            self.hier_levels_idx = None

        # Bookkeeping for nice error messages on unknown species names.
        self._known_species: List[str] = list(self.species_to_sp_idx.keys())

    # ------------------------------------------------------------------
    def _names_to_sp_idx(self, names: List[str], device: torch.device) -> torch.Tensor:
        """Map a python list of names to a LongTensor[B] of sp_idx rows."""
        rows = []
        unknown = []
        for n in names:
            if n in self.species_to_sp_idx:
                rows.append(self.species_to_sp_idx[n])
            else:
                unknown.append(n)
                rows.append(-1)
        if unknown:
            raise KeyError(
                f"{len(unknown)} species not present in the taxonomy graph "
                f"(first few: {unknown[:5]}). Rebuild the graph with these CSVs included."
            )
        return torch.tensor(rows, dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    def forward(self, species_ids_or_names) -> torch.Tensor:
        """Return ``[B, out_dim]`` species feature.

        ``species_ids_or_names`` may be either:
          - a list/tuple of species names (str), or
          - a LongTensor of sp_idx rows (precomputed by the data loader).
        """
        all_node_emb = self.gnn()                          # [N, out_dim]

        if isinstance(species_ids_or_names, torch.Tensor):
            sp_idx = species_ids_or_names.long().to(all_node_emb.device)
        else:
            sp_idx = self._names_to_sp_idx(list(species_ids_or_names), all_node_emb.device)

        if self.fusion == "leaf":
            # Take the species (rank 7) ancestor entry; this is always >=0
            # because every leaf in our graph is a species-or-finer node.
            leaf_node = self.ancestors_per_species[sp_idx, self.LEVEL_TO_IDX["species"]]
            return all_node_emb[leaf_node]                 # [B, out_dim]

        # fusion == "hier" --------------------------------------------------
        # For each requested rank, gather the ancestor node id; if -1, fall
        # back to the rank-specific learnable "absent" vector.
        chunks = []
        for k, lvl_idx in enumerate(self.hier_levels_idx):
            anc = self.ancestors_per_species[sp_idx, lvl_idx]      # [B], may be -1
            present = anc.ge(0)
            safe_idx = anc.clamp(min=0)
            gathered = all_node_emb[safe_idx]                      # [B, out_dim]
            absent = self.absent_emb[k].unsqueeze(0).expand_as(gathered)
            chunks.append(torch.where(present.unsqueeze(-1), gathered, absent))
        cat = torch.cat(chunks, dim=-1)                            # [B, out_dim*L]
        return self.hier_proj(cat)                                 # [B, out_dim]


# ---------------------------------------------------------------------------
# Convenience: load a saved graph dict and instantiate the encoder.
# ---------------------------------------------------------------------------
def build_species_encoder_from_graph(
    graph_path: str,
    *,
    hidden: int = 128,
    out_dim: int = 64,
    num_layers: int = 2,
    gnn_type: str = "gcn",
    heads: int = 4,
    dropout: float = 0.1,
    fusion: str = "leaf",
    freeze_init: bool = True,
) -> TaxonomySpeciesEncoder:
    g = torch.load(graph_path, weights_only=False, map_location="cpu")
    species_to_sp_idx = {name: i for i, name in enumerate(g["species_names"])}
    return TaxonomySpeciesEncoder(
        init_features=g["init_features"],
        edge_index=g["edge_index"],
        ancestors_per_species=g["ancestors_per_species"],
        species_to_sp_idx=species_to_sp_idx,
        in_dim=g["init_features"].size(1),
        hidden=hidden,
        out_dim=out_dim,
        num_layers=num_layers,
        gnn_type=gnn_type,
        heads=heads,
        dropout=dropout,
        fusion=fusion,
        freeze_init=freeze_init,
    )
