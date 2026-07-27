"""TaxonomyGNN: encode the (frozen) NCBI canonical-rank lineage DAG.

The forward pass takes the static 768-d initial node features built by
``build_taxonomy_graph.py``, runs ``num_layers`` of GCN or GAT message-passing
over the (parent->child + reverse) edges, and returns the per-node embedding
projected to ``out_dim``.

Four fusion strategies for going from per-node embeddings to a per-species
batch embedding:
    fusion='leaf'     : take only the species (leaf) node embedding.
    fusion='hier'     : concat over a list of canonical-rank ancestors,
                        then Linear -> out_dim. Levels controlled by
                        ``hier_levels`` (default ('species','genus','family')).
    fusion='hier_attn': attention-pool over the chosen levels (let the
                        model self-weight species/genus/family/...).
    fusion='hier_raw' : concat over the hier_levels (NO projection back to
                        out_dim). Output width is out_dim * len(hier_levels),
                        i.e. exposes the raw per-level embeddings to the
                        downstream head. ``self.out_dim`` is overridden to
                        the resulting width.

Regardless of fusion mode (as long as hier_levels are valid), a separate
``forward_levels(species_ids_or_names)`` method returns the raw
``[B, len(hier_levels), out_dim_per_node]`` tensor (before any hier_proj /
hier_attn / hier_raw reshape). This is what the prefix-injection path in
``plm_models.py`` consumes to build the 3 species prefix tokens.

Optional knobs (Stage 0 plan):
    use_lora_init=True : keep the 768-d PubMedBERT init *frozen* and add a
                         low-rank residual ``A @ B`` (A in [N, r], B in [r, 768])
                         that IS trainable. r=lora_rank (default 16).
                         If ``freeze_init=False`` instead, the full init is
                         a Parameter (high capacity, high overfit risk).
    use_residual=True  : add skip connection in each GCN layer.
    use_layernorm=True : apply LayerNorm before ReLU between GCN layers.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, GCNConv

from PLM_head import SpeciesAdapter


HIER_LEVELS_FULL = ("domain", "kingdom", "phylum", "class",
                    "order",  "family",  "genus",  "species")

LEVEL_TO_IDX = {name: i for i, name in enumerate(HIER_LEVELS_FULL)}


# ---------------------------------------------------------------------------
# The taxonomy GNN itself.
# ---------------------------------------------------------------------------
class TaxonomyGNN(nn.Module):
    """Encode the taxonomy DAG; output ``[N_nodes, out_dim]``.

    See module docstring for the full list of options.
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
        use_lora_init: bool = False,
        lora_rank: int = 16,
        use_residual: bool = False,
        use_layernorm: bool = False,
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
        self.use_lora_init = use_lora_init
        self.lora_rank = lora_rank
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm

        N = init_features.size(0)

        # ------ static node features ------------------------------------
        # Three init-feature regimes:
        #  (1) freeze_init=True, use_lora_init=False  -> buffer (no params)
        #  (2) freeze_init=True, use_lora_init=True   -> buffer + (A,B) low-rank residual
        #  (3) freeze_init=False                       -> full Parameter
        if freeze_init:
            self.register_buffer("init_features", init_features.float(), persistent=False)
            if use_lora_init:
                self.lora_A = nn.Parameter(torch.zeros(N, lora_rank))
                self.lora_B = nn.Parameter(torch.zeros(lora_rank, in_dim))
                # Conventional LoRA init: A small Gaussian, B zero.
                nn.init.normal_(self.lora_A, std=0.02)
                nn.init.zeros_(self.lora_B)
            else:
                self.lora_A = None
                self.lora_B = None
        else:
            assert not use_lora_init, "use_lora_init only valid with freeze_init=True"
            self.init_features = nn.Parameter(init_features.float().clone())
            self.lora_A = None
            self.lora_B = None

        self.register_buffer("edge_index", edge_index.long(), persistent=False)

        self.input_proj = nn.Linear(in_dim, hidden)

        # GNN layers
        self.layers = nn.ModuleList()
        if gnn_type == "gcn":
            for _ in range(num_layers):
                self.layers.append(GCNConv(hidden, hidden))
        elif gnn_type == "gat":
            for _ in range(num_layers):
                self.layers.append(
                    GATConv(hidden, hidden, heads=heads, concat=False, dropout=dropout)
                )
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type}")

        if use_layernorm:
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        else:
            self.norms = None

        self.output_proj = nn.Linear(hidden, out_dim)

    # ------------------------------------------------------------------
    def _resolve_init(self) -> torch.Tensor:
        """Apply LoRA residual on top of frozen init if enabled, else just init."""
        if self.use_lora_init and self.lora_A is not None:
            # init_features is a buffer (frozen) but A,B are trainable.
            return self.init_features + self.lora_A @ self.lora_B
        return self.init_features

    # ------------------------------------------------------------------
    def forward(self) -> torch.Tensor:
        """Return all-node embeddings ``[N, out_dim]``."""
        x = self.input_proj(self._resolve_init())                 # [N, hidden]
        for i, layer in enumerate(self.layers):
            h = layer(x, self.edge_index)                         # [N, hidden]
            if self.use_residual and h.shape == x.shape:
                h = h + x
            if self.norms is not None:
                h = self.norms[i](h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                if self.dropout > 0:
                    h = F.dropout(h, p=self.dropout, training=self.training)
            x = h
        return self.output_proj(x)                                # [N, out_dim]


# ---------------------------------------------------------------------------
# End-to-end species channel: GNN + (leaf | hier | hier_attn) fusion.
# ---------------------------------------------------------------------------
class TaxonomySpeciesEncoder(nn.Module):
    """Wraps TaxonomyGNN with one of three rank-aggregation strategies.

    fusion='leaf'     -> species_emb only                       (F1 in plan)
    fusion='hier'     -> concat(hier_levels) + Linear           (F2 in plan)
    fusion='hier_attn'-> attention-pool over hier_levels        (Stage 1 stretch)

    Output is always ``[B, out_dim]`` so swapping fusion does not require
    touching the downstream MLP / fusion-with-ESM modules.
    """

    LEVEL_TO_IDX = LEVEL_TO_IDX

    def __init__(
        self,
        init_features: torch.Tensor,
        edge_index: torch.Tensor,
        ancestors_per_species: torch.Tensor,    # [S, 8]
        species_to_sp_idx: Dict[str, int],
        in_dim: int = 768,
        hidden: int = 128,
        out_dim: int = 64,
        num_layers: int = 2,
        gnn_type: str = "gcn",
        heads: int = 4,
        dropout: float = 0.1,
        fusion: Literal["leaf", "hier", "hier_attn", "hier_raw", "gate_hard"] = "leaf",
        hier_levels: Sequence[str] = ("species", "genus", "family"),
        freeze_init: bool = True,
        use_lora_init: bool = False,
        lora_rank: int = 16,
        use_residual: bool = False,
        use_layernorm: bool = False,
        attn_heads: int = 4,
        # ----- gate_hard fusion knobs -----------------------------------
        # Per-species train sample counts (name -> count). Species with
        # count >= gate_count_threshold use the adapter-identical passthrough
        # branch (no message passing, leaf init only); the rest use the GCN
        # aggregated leaf embedding.
        species_train_counts: Optional[Dict[str, int]] = None,
        gate_count_threshold: int = 100,
        adapter_bottleneck: int = 128,
        adapter_dropout: float = 0.1,
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
            use_lora_init=use_lora_init,
            lora_rank=lora_rank,
            use_residual=use_residual,
            use_layernorm=use_layernorm,
        )

        self.register_buffer("ancestors_per_species", ancestors_per_species.long(), persistent=False)
        self.species_to_sp_idx: Dict[str, int] = dict(species_to_sp_idx)

        if fusion not in ("leaf", "hier", "hier_attn", "hier_raw", "gate_hard"):
            raise ValueError(f"Unknown fusion mode: {fusion!r}")
        self.fusion = fusion
        self.gate_count_threshold = int(gate_count_threshold)
        # self.out_dim is the width of the tensor returned by ``forward``.
        # For ``hier_raw`` it is out_dim * len(hier_levels); for the others
        # it equals out_dim. We also record the per-node width so that
        # ``forward_levels`` (which returns the raw [B, L, out_dim] before any
        # cross-level pooling) is always well-defined.
        self.per_node_out_dim = out_dim
        self.out_dim = out_dim

        # Validate hier_levels and stash indices.
        if fusion in ("hier", "hier_attn", "hier_raw"):
            for lvl in hier_levels:
                if lvl not in self.LEVEL_TO_IDX:
                    raise ValueError(f"Unknown taxonomic level: {lvl!r}")
            self.hier_levels: Tuple[str, ...] = tuple(hier_levels)
            self.hier_levels_idx: List[int] = [self.LEVEL_TO_IDX[l] for l in self.hier_levels]

            # Per-level "absent" embedding (used when ancestors_per_species[i,k] == -1)
            self.absent_emb = nn.Parameter(torch.zeros(len(self.hier_levels), out_dim))
            nn.init.normal_(self.absent_emb, std=0.02)

            if fusion == "hier":
                self.hier_proj = nn.Linear(out_dim * len(self.hier_levels), out_dim)
                self.attn = None
            elif fusion == "hier_attn":
                self.hier_proj = None
                # MultiheadAttention over the level dim. Query is a learnable
                # token; keys/values are the gathered per-level embeddings.
                self.query_token = nn.Parameter(torch.zeros(1, 1, out_dim))
                nn.init.normal_(self.query_token, std=0.02)
                self.attn = nn.MultiheadAttention(
                    embed_dim=out_dim, num_heads=attn_heads,
                    dropout=dropout, batch_first=True,
                )
            else:  # hier_raw: no cross-level projection at all.
                self.hier_proj = None
                self.attn = None
                # Output width that downstream modules see.
                self.out_dim = out_dim * len(self.hier_levels)
        else:
            self.hier_levels = ()
            self.hier_levels_idx = []
            self.absent_emb = None
            self.hier_proj = None
            self.attn = None

        # ----- gate_hard: adapter-identical passthrough branch --------------
        # High-count (count >= threshold) species bypass message passing and
        # go through a SpeciesAdapter that is architecturally identical to the
        # 'adapter' species channel (Linear->LN->GELU->Dropout->Linear->GELU),
        # fed the SAME 768-d PubMedBERT leaf init vector. Low-count species use
        # the GCN-aggregated leaf embedding. Both output ``out_dim`` so the
        # downstream head sees a fixed width.
        if fusion == "gate_hard":
            self.passthrough_adapter = SpeciesAdapter(
                in_dim=in_dim,
                bottleneck=adapter_bottleneck,
                out_dim=out_dim,
                dropout=adapter_dropout,
            )
            # Per-species passthrough mask, indexed by sp_idx (species_names order).
            S = len(self.species_to_sp_idx)
            mask = torch.zeros(S, dtype=torch.bool)
            if species_train_counts is not None:
                for name, c in species_train_counts.items():
                    j = self.species_to_sp_idx.get(name)
                    if j is not None and int(c) >= self.gate_count_threshold:
                        mask[j] = True
            self.register_buffer("passthrough_mask", mask, persistent=True)
        else:
            self.passthrough_adapter = None
            self.register_buffer("passthrough_mask", None, persistent=False)

        self._known_species: List[str] = list(self.species_to_sp_idx.keys())

    # ------------------------------------------------------------------
    def set_species_train_counts(self, species_train_counts: Dict[str, int]) -> None:
        """(Re)build the passthrough mask from a {species_name -> count} dict.

        Used when counts are only known at train time (split-dependent) while
        the encoder was constructed from the split-agnostic graph.
        """
        if self.fusion != "gate_hard":
            raise RuntimeError("set_species_train_counts() only valid for fusion='gate_hard'.")
        S = len(self.species_to_sp_idx)
        mask = torch.zeros(S, dtype=torch.bool)
        for name, c in species_train_counts.items():
            j = self.species_to_sp_idx.get(name)
            if j is not None and int(c) >= self.gate_count_threshold:
                mask[j] = True
        self.passthrough_mask = mask.to(self.passthrough_mask.device)
        n_pass = int(mask.sum())
        print(f"[TaxonomySpeciesEncoder] gate_hard: {n_pass}/{S} species "
              f"passthrough (count >= {self.gate_count_threshold}), "
              f"{S - n_pass} aggregated via GCN.")

    # ------------------------------------------------------------------
    def _names_to_sp_idx(self, names, device: torch.device) -> torch.Tensor:
        rows, unknown = [], []
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
    def _gather_level_embs(self, all_node_emb: torch.Tensor,
                           sp_idx: torch.Tensor) -> torch.Tensor:
        """Return [B, L, out_dim] (with absent vectors filled in)."""
        chunks = []
        for k, lvl_idx in enumerate(self.hier_levels_idx):
            anc = self.ancestors_per_species[sp_idx, lvl_idx]      # [B], may be -1
            present = anc.ge(0)
            safe_idx = anc.clamp(min=0)
            gathered = all_node_emb[safe_idx]                      # [B, out_dim]
            absent = self.absent_emb[k].unsqueeze(0).expand_as(gathered)
            chunks.append(torch.where(present.unsqueeze(-1), gathered, absent))
        return torch.stack(chunks, dim=1)                          # [B, L, out_dim]

    # ------------------------------------------------------------------
    def _resolve_sp_idx(self, species_ids_or_names, device: torch.device) -> torch.Tensor:
        if isinstance(species_ids_or_names, torch.Tensor):
            return species_ids_or_names.long().to(device)
        return self._names_to_sp_idx(list(species_ids_or_names), device)

    def forward(self, species_ids_or_names) -> torch.Tensor:
        all_node_emb = self.gnn()                                  # [N, per_node_out_dim]
        sp_idx = self._resolve_sp_idx(species_ids_or_names, all_node_emb.device)

        if self.fusion == "leaf":
            leaf_node = self.ancestors_per_species[sp_idx, self.LEVEL_TO_IDX["species"]]
            return all_node_emb[leaf_node]                         # [B, out_dim]

        if self.fusion == "gate_hard":
            
            leaf_node = self.ancestors_per_species[sp_idx, self.LEVEL_TO_IDX["species"]]
            z_agg = all_node_emb[leaf_node]                        # [B, out_dim] (GCN)
            leaf_init = self.gnn._resolve_init()[leaf_node]        # [B, in_dim] raw PubMedBERT
            z_pass = self.passthrough_adapter(leaf_init)           # [B, out_dim] (adapter-identical)
            gate = self.passthrough_mask[sp_idx].unsqueeze(-1)     # [B, 1] True=passthrough
            return torch.where(gate, z_pass, z_agg)                # [B, out_dim]

        # hier / hier_attn / hier_raw all gather per-level embs first ---
        level_embs = self._gather_level_embs(all_node_emb, sp_idx)  # [B, L, out_dim]
        B = level_embs.size(0)

        if self.fusion == "hier":
            return self.hier_proj(level_embs.reshape(B, -1))        # [B, out_dim]

        if self.fusion == "hier_raw":
            # No cross-level projection; expose the concatenated raw vector.
            return level_embs.reshape(B, -1)                        # [B, L * out_dim]

        # hier_attn ----------------------------------------------------
        q = self.query_token.expand(B, -1, -1)                      # [B, 1, out_dim]
        out, _ = self.attn(q, level_embs, level_embs, need_weights=False)
        return out.squeeze(1)                                       # [B, out_dim]

    # ------------------------------------------------------------------
    def forward_levels(self, species_ids_or_names) -> torch.Tensor:
        """Return the raw per-level embeddings ``[B, len(hier_levels), per_node_out_dim]``.

        Bypasses ``hier_proj`` / ``hier_attn`` / ``hier_raw`` reshape. This is
        what the prefix-injection path consumes to feed N species tokens into
        the ESM input. Requires fusion in {hier, hier_attn, hier_raw}.
        """
        if self.fusion not in ("hier", "hier_attn", "hier_raw"):
            raise RuntimeError(
                f"forward_levels() requires fusion in {{hier, hier_attn, hier_raw}}; "
                f"got fusion={self.fusion!r}."
            )
        all_node_emb = self.gnn()                                  # [N, per_node_out_dim]
        sp_idx = self._resolve_sp_idx(species_ids_or_names, all_node_emb.device)
        return self._gather_level_embs(all_node_emb, sp_idx)        # [B, L, per_node_out_dim]


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
    fusion: str = "leaf",   # leaf | hier | hier_attn | hier_raw
    hier_levels: Sequence[str] = ("species", "genus", "family"),
    freeze_init: bool = True,
    use_lora_init: bool = False,
    lora_rank: int = 16,
    use_residual: bool = False,
    use_layernorm: bool = False,
    attn_heads: int = 4,
    species_train_counts: Optional[Dict[str, int]] = None,
    gate_count_threshold: int = 100,
    adapter_bottleneck: int = 128,
    adapter_dropout: float = 0.1,
    random_init: bool = False,
    random_init_seed: Optional[int] = None,
    random_init_std: Optional[float] = None,
) -> TaxonomySpeciesEncoder:
    g = torch.load(graph_path, weights_only=False, map_location="cpu")
    species_to_sp_idx = {name: i for i, name in enumerate(g["species_names"])}

    init_features = g["init_features"]
    if random_init:
        # Baseline control: destroy the PubMedBERT semantics by replacing every
        # node's 768-d feature with a random Gaussian vector, scaled to match the
        # std of the real features so the downstream GNN sees comparable
        # magnitudes. Everything else (graph topology, GCN, fusion, freeze_init)
        # stays identical to v15, isolating the value of the pretrained text
        # embedding. The draw is seeded so runs are reproducible and vary per seed.
        n_nodes, in_dim = init_features.shape
        std = (random_init_std if random_init_std is not None
               else float(init_features.float().std()))
        gen = torch.Generator()
        if random_init_seed is not None:
            gen.manual_seed(int(random_init_seed))
        init_features = torch.randn(n_nodes, in_dim, generator=gen) * std
        print(f"[build_species_encoder] RANDOM node features: "
              f"shape=({n_nodes},{in_dim}) std={std:.4f} seed={random_init_seed}")

    return TaxonomySpeciesEncoder(
        init_features=init_features,
        edge_index=g["edge_index"],
        ancestors_per_species=g["ancestors_per_species"],
        species_to_sp_idx=species_to_sp_idx,
        in_dim=init_features.size(1),
        hidden=hidden,
        out_dim=out_dim,
        num_layers=num_layers,
        gnn_type=gnn_type,
        heads=heads,
        dropout=dropout,
        fusion=fusion,
        hier_levels=hier_levels,
        freeze_init=freeze_init,
        use_lora_init=use_lora_init,
        lora_rank=lora_rank,
        use_residual=use_residual,
        use_layernorm=use_layernorm,
        attn_heads=attn_heads,
        species_train_counts=species_train_counts,
        gate_count_threshold=gate_count_threshold,
        adapter_bottleneck=adapter_bottleneck,
        adapter_dropout=adapter_dropout,
    )
