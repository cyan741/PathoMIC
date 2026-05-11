import pdb
import pickle
import torch
import pandas as pd
import numpy as np
import random
import os
from typing import List, Optional, Dict, Tuple
from torch.utils.data import Dataset, DataLoader

from losses import compute_lds_weights, assign_bin


def load_species_embeddings(pkl_path: str) -> Dict[str, torch.Tensor]:
    """Load the species embedding pkl (list of dicts) -> {pathogen -> Tensor}."""
    with open(pkl_path, "rb") as f:
        records = pickle.load(f)
    mapping = {}
    for r in records:
        emb = r["embedding"]
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb, dtype=torch.float32)
        else:
            emb = emb.float()
        mapping[r["pathogen"]] = emb
    return mapping


def load_species_to_node_id(graph_path: str) -> Dict[str, int]:
    """{Target_Species -> sp_idx}."""
    g = torch.load(graph_path, weights_only=False, map_location="cpu")
    return {name: i for i, name in enumerate(g["species_names"])}


# ---------------------------------------------------------------------------
# Bucketing helper for GroupDRO ``--loss_dro_group_by bucket``.
# ---------------------------------------------------------------------------
DEFAULT_BUCKET_BOUNDS = (0, 5, 20, 100, float("inf"))


def species_bucket_id(name: str, sp_count: Dict[str, int],
                      bounds: Tuple[float, ...] = DEFAULT_BUCKET_BOUNDS) -> int:
    n = sp_count.get(str(name), 0)
    for i in range(len(bounds) - 1):
        if bounds[i] <= n < bounds[i + 1]:
            return i
    return len(bounds) - 2


def num_buckets(bounds: Tuple[float, ...] = DEFAULT_BUCKET_BOUNDS) -> int:
    return len(bounds) - 1


# ---------------------------------------------------------------------------
class MIC_Dataset(Dataset):
    """Returns variable-length tuples; the training loop unpacks based on
    species_mode + (whether per-sample meta is needed for loss).
    """

    def __init__(
        self,
        mic_df: pd.DataFrame,
        max_length: int = 70,
        species_mode: str = "none",
        species_emb_map: Optional[Dict[str, torch.Tensor]] = None,
        species_emb_dim: int = 768,
        species_to_node_id: Optional[Dict[str, int]] = None,
        unknown_species_node_id: int = -1,
        # ----- loss meta -------------------------------------------------
        return_meta: bool = False,
        bin_edges: Optional[np.ndarray] = None,         # for LDS
        bucket_lookup: Optional[Dict[str, int]] = None, # for GroupDRO ``bucket`` mode
        sp_to_group_id: Optional[Dict[str, int]] = None # for GroupDRO ``species`` mode
    ):
        if species_mode not in ("none", "adapter", "gnn", "both"):
            raise ValueError(f"Unknown species_mode: {species_mode!r}")
        self.species_mode = species_mode
        self.amp_seqs = mic_df.Sequence.tolist()
        self.mic_values = torch.tensor(mic_df.Median_MIC, dtype=torch.float32)
        self.max_length = max_length
        self.return_meta = return_meta

        needs_emb = species_mode in ("adapter", "both")
        needs_node = species_mode in ("gnn", "both")

        if needs_emb or needs_node or return_meta:
            if "Target_Species" not in mic_df.columns:
                raise KeyError(
                    f"species_mode={species_mode!r} (or return_meta=True) requires a "
                    f"`Target_Species` column in the dataframe."
                )
            self.species_names = mic_df.Target_Species.astype(str).tolist()
        else:
            self.species_names = None

        # legacy adapter pathway
        self.species_emb_map = species_emb_map if needs_emb else None
        self.species_emb_dim = species_emb_dim
        self._zero_emb = (
            torch.zeros(species_emb_dim, dtype=torch.float32) if needs_emb else None
        )
        if needs_emb and self.species_emb_map is None:
            raise ValueError(f"species_mode={species_mode!r} requires species_emb_map.")

        # GNN pathway
        self.species_to_node_id = species_to_node_id if needs_node else None
        self._unknown_node_id = unknown_species_node_id
        if needs_node and self.species_to_node_id is None:
            raise ValueError(f"species_mode={species_mode!r} requires species_to_node_id.")
        if needs_node:
            missing = [n for n in self.species_names if n not in self.species_to_node_id]
            if missing:
                raise KeyError(
                    f"{len(missing)} species are not present in the taxonomy graph "
                    f"(first few: {missing[:5]}). Rebuild taxonomy_graph.pt."
                )

        # ---- meta --------------------------------------------------------
        self.bin_edges = bin_edges
        self._bin_idx = (
            torch.tensor(assign_bin(mic_df["Median_MIC"].values, bin_edges),
                         dtype=torch.long)
            if (return_meta and bin_edges is not None) else None
        )
        self.bucket_lookup = bucket_lookup
        self.sp_to_group_id = sp_to_group_id
        if return_meta and bucket_lookup is not None:
            # Pre-compute bucket id for every row using the species_count_lookup.
            self._bucket_id = torch.tensor(
                [bucket_lookup[str(n)] for n in self.species_names],
                dtype=torch.long,
            )
        else:
            self._bucket_id = None
        if return_meta and sp_to_group_id is not None:
            self._species_group_id = torch.tensor(
                [sp_to_group_id.get(str(n), 0) for n in self.species_names],
                dtype=torch.long,
            )
        else:
            self._species_group_id = None

    def __len__(self):
        return len(self.amp_seqs)

    def _get_meta(self, idx) -> Dict[str, torch.Tensor]:
        meta = {}
        if self._bin_idx is not None:
            meta["bin_idx"] = self._bin_idx[idx]
        if self._bucket_id is not None:
            meta["bucket_id"] = self._bucket_id[idx]
        if self._species_group_id is not None:
            meta["species_group_id"] = self._species_group_id[idx]
        return meta

    def __getitem__(self, idx):
        sequence = self.amp_seqs[idx].ljust(self.max_length, "X")
        mic_value = self.mic_values[idx]
        mode = self.species_mode

        if not self.return_meta:
            # Backwards-compatible tuples (no meta).
            if mode == "none":
                return sequence, mic_value
            sp_name = self.species_names[idx]
            if mode == "adapter":
                sp_emb = self.species_emb_map.get(sp_name, self._zero_emb)
                return sequence, sp_emb, mic_value
            if mode == "gnn":
                node_id = torch.tensor(
                    self.species_to_node_id.get(sp_name, self._unknown_node_id),
                    dtype=torch.long,
                )
                return sequence, node_id, mic_value
            sp_emb = self.species_emb_map.get(sp_name, self._zero_emb)
            node_id = torch.tensor(
                self.species_to_node_id.get(sp_name, self._unknown_node_id),
                dtype=torch.long,
            )
            return sequence, sp_emb, node_id, mic_value

        # With meta: append a dict of LongTensors at the end.
        meta = self._get_meta(idx)
        if mode == "none":
            return sequence, mic_value, meta
        sp_name = self.species_names[idx]
        if mode == "adapter":
            sp_emb = self.species_emb_map.get(sp_name, self._zero_emb)
            return sequence, sp_emb, mic_value, meta
        if mode == "gnn":
            node_id = torch.tensor(
                self.species_to_node_id.get(sp_name, self._unknown_node_id),
                dtype=torch.long,
            )
            return sequence, node_id, mic_value, meta
        sp_emb = self.species_emb_map.get(sp_name, self._zero_emb)
        node_id = torch.tensor(
            self.species_to_node_id.get(sp_name, self._unknown_node_id),
            dtype=torch.long,
        )
        return sequence, sp_emb, node_id, mic_value, meta


def seq2token(sequences: List[str], tokenizer, device) -> torch.Tensor:
    token_list = []
    for sequence in sequences:
        tokens = tokenizer.encode(sequence)
        token_list.append(tokens)
    token_tensor = torch.tensor(token_list)
    token_tensor = token_tensor.to(device)
    return token_tensor


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
def _meta_collate(batch):
    """Default collate but stack each meta-key separately into a dict.

    Returns a tuple ``(*head_items, meta_dict)``. ``default_collate`` returns
    a *list* (not tuple) when each sample is a tuple, so we always splat it.
    """
    if isinstance(batch[0][-1], dict):
        meta_keys = batch[0][-1].keys()
        head = [b[:-1] for b in batch]
        rest = torch.utils.data.default_collate(head)
        meta = {k: torch.stack([b[-1][k] for b in batch], dim=0) for k in meta_keys}
        if isinstance(rest, (list, tuple)):
            return (*rest, meta)
        return (rest, meta)
    return torch.utils.data.default_collate(batch)


def data_loader(data_path, batch_size, num_workers, seed,
                species_mode: str = "none",
                species_emb_path: Optional[str] = None,
                species_emb_dim: int = 768,
                taxo_graph_path: Optional[str] = None,
                # ----- loss meta options -----
                loss_type: str = "mse",
                lds_num_bins: int = 50,
                lds_sigma: float = 2.0,
                dro_group_by: str = "bucket",
                dro_bucket_bounds: Tuple[float, ...] = DEFAULT_BUCKET_BOUNDS):
    """Build (train_loader, val_loader, test_loader).

    If ``loss_type`` requires per-sample meta (lds, group_dro), we precompute
    bin assignments / bucket ids on the TRAIN dataframe and embed them in the
    Dataset so they round-trip through DataLoader properly.
    """
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    val_df   = pd.read_csv(os.path.join(data_path, "val.csv"))
    test_df  = pd.read_csv(os.path.join(data_path, "test.csv"))

    if species_mode == "none" and species_emb_path is not None:
        species_mode = "adapter"

    species_emb_map = None
    if species_mode in ("adapter", "both"):
        if species_emb_path is None:
            raise ValueError(f"species_mode={species_mode!r} but species_emb_path is None.")
        species_emb_map = load_species_embeddings(species_emb_path)
        print(f"[data_loader] loaded {len(species_emb_map)} species embeddings "
              f"(dim={species_emb_dim}) from {species_emb_path}")

    species_to_node_id = None
    if species_mode in ("gnn", "both"):
        if taxo_graph_path is None:
            raise ValueError(f"species_mode={species_mode!r} but taxo_graph_path is None.")
        species_to_node_id = load_species_to_node_id(taxo_graph_path)
        print(f"[data_loader] loaded {len(species_to_node_id)} species->node_id mappings "
              f"from {taxo_graph_path}")

    # ---------- decide which meta we need --------------------------------
    needs_meta = loss_type.lower() in ("lds", "group_dro")
    bin_edges = None
    bin_weight = None
    bucket_lookup = None
    sp_to_group_id = None
    dro_num_groups = None

    if needs_meta:
        if loss_type.lower() == "lds":
            bin_edges, bin_weight = compute_lds_weights(
                train_df["Median_MIC"].values,
                num_bins=lds_num_bins, sigma=lds_sigma,
            )
            print(f"[data_loader] LDS: {lds_num_bins} bins, sigma={lds_sigma}, "
                  f"weight range=({bin_weight.min():.3f}, {bin_weight.max():.3f})")
        elif loss_type.lower() == "group_dro":
            sp_count = train_df["Target_Species"].astype(str).value_counts().to_dict()
            if dro_group_by == "bucket":
                bucket_lookup = {n: species_bucket_id(n, sp_count, dro_bucket_bounds)
                                 for n in sp_count.keys()}
                # Also include species in val/test that may not be in train.
                for df in (val_df, test_df):
                    for n in df["Target_Species"].astype(str).unique():
                        bucket_lookup.setdefault(n, species_bucket_id(n, sp_count, dro_bucket_bounds))
                dro_num_groups = num_buckets(dro_bucket_bounds)
                print(f"[data_loader] GroupDRO ({dro_group_by}): "
                      f"{dro_num_groups} buckets, bounds={dro_bucket_bounds}")
            elif dro_group_by == "species":
                # Each species is its own group.
                all_species = set(train_df["Target_Species"].astype(str).unique())
                sp_to_group_id = {n: i for i, n in enumerate(sorted(all_species))}
                # val/test species not in train: assign to group 0 (will get 0 weight).
                dro_num_groups = len(sp_to_group_id)
                print(f"[data_loader] GroupDRO ({dro_group_by}): {dro_num_groups} species groups")
            else:
                raise ValueError(f"Unknown dro_group_by: {dro_group_by!r}")

    common_kwargs = dict(
        species_mode=species_mode,
        species_emb_map=species_emb_map,
        species_emb_dim=species_emb_dim,
        species_to_node_id=species_to_node_id,
        return_meta=needs_meta,
        bin_edges=bin_edges,
        bucket_lookup=bucket_lookup,
        sp_to_group_id=sp_to_group_id,
    )

    train_dataset = MIC_Dataset(train_df, **common_kwargs)
    val_dataset   = MIC_Dataset(val_df,   **common_kwargs)
    test_dataset  = MIC_Dataset(test_df,  **common_kwargs)

    collate_fn = _meta_collate if needs_meta else None
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, worker_init_fn=seed_worker,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, worker_init_fn=seed_worker,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, worker_init_fn=seed_worker,
        collate_fn=collate_fn,
    )

    # Stash global loss meta on the train_loader for the trainer to pick up.
    train_loader.loss_meta = {
        "needs_meta": needs_meta,
        "bin_edges": bin_edges,
        "bin_weight": (torch.tensor(bin_weight) if bin_weight is not None else None),
        "dro_num_groups": dro_num_groups,
        "dro_group_by": dro_group_by,
    }
    return train_loader, val_loader, test_loader
