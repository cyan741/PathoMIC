import pdb
import pickle
import torch
import pandas as pd
import numpy as np
import random
import os
from typing import List, Optional, Dict
from torch.utils.data import Dataset, DataLoader
# load AA_seq + MIC_value  
# tokenize AA_seq put it to training step


def load_species_embeddings(pkl_path: str) -> Dict[str, torch.Tensor]:
    """
    Load the species embedding pkl (list of dicts with keys
    ['pathogen', 'types', 'embedding', 'source']) and return a
    {pathogen_name -> FloatTensor[emb_dim]} mapping.
    """
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
    """Load {Target_Species -> sp_idx (row in ancestors_per_species)} from the
    pre-built taxonomy graph .pt file.
    """
    g = torch.load(graph_path, weights_only=False, map_location="cpu")
    return {name: i for i, name in enumerate(g["species_names"])}


class MIC_Dataset(Dataset):
    """Dataset that supports four species injection modes.

    species_mode = 'none'    -> returns (sequence, mic)
                 = 'adapter' -> returns (sequence, species_emb_768d, mic)
                 = 'gnn'     -> returns (sequence, sp_node_id (long), mic)
                 = 'both'    -> returns (sequence, species_emb_768d, sp_node_id, mic)

    The legacy ``species_emb_map`` (PubMedBERT 768-d per species) is still
    consumed by ``adapter`` / ``both`` modes; the new ``species_to_node_id`` map
    is consumed by ``gnn`` / ``both`` modes.
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
    ):
        if species_mode not in ("none", "adapter", "gnn", "both"):
            raise ValueError(f"Unknown species_mode: {species_mode!r}")
        self.species_mode = species_mode
        self.amp_seqs = mic_df.Sequence.tolist()
        self.mic_values = torch.tensor(mic_df.Median_MIC, dtype=torch.float32)
        self.max_length = max_length

        needs_emb = species_mode in ("adapter", "both")
        needs_node = species_mode in ("gnn", "both")

        if needs_emb or needs_node:
            if "Target_Species" not in mic_df.columns:
                raise KeyError(
                    f"species_mode={species_mode!r} requires a `Target_Species` "
                    f"column in the dataframe."
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
            raise ValueError(
                f"species_mode={species_mode!r} requires species_emb_map."
            )

        # new GNN pathway
        self.species_to_node_id = species_to_node_id if needs_node else None
        self._unknown_node_id = unknown_species_node_id
        if needs_node and self.species_to_node_id is None:
            raise ValueError(
                f"species_mode={species_mode!r} requires species_to_node_id."
            )
        if needs_node:
            # Eagerly resolve names to ids so we error out at construction-time
            # for any typo, instead of in the middle of training.
            missing = [n for n in self.species_names if n not in self.species_to_node_id]
            if missing:
                raise KeyError(
                    f"{len(missing)} species are not present in the taxonomy graph "
                    f"(first few: {missing[:5]}). Rebuild taxonomy_graph.pt."
                )

    def __len__(self):
        return len(self.amp_seqs)

    def __getitem__(self, idx):
        sequence = self.amp_seqs[idx].ljust(self.max_length, "X")
        mic_value = self.mic_values[idx]

        mode = self.species_mode
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
        # both
        sp_emb = self.species_emb_map.get(sp_name, self._zero_emb)
        node_id = torch.tensor(
            self.species_to_node_id.get(sp_name, self._unknown_node_id),
            dtype=torch.long,
        )
        return sequence, sp_emb, node_id, mic_value


def seq2token(sequences:List[str], tokenizer, device) -> torch.Tensor:
    # tokenize sequences and turn to input_ids tensor
    token_list = []
    for sequence in sequences:
        tokens = tokenizer.encode(sequence)
        token_list.append(tokens)
    token_tensor = torch.tensor(token_list)
    token_tensor = token_tensor.to(device)
    # pdb.set_trace()
    # print("shape of input_ids :", token_tensor.shape) # (batch_size, seq_len+2)
    return token_tensor

# Dataloder seed
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def data_loader(data_path, batch_size, num_workers, seed,
                species_mode: str = "none",
                species_emb_path: Optional[str] = None,
                species_emb_dim: int = 768,
                taxo_graph_path: Optional[str] = None):
    """Return train/val/test data loaders, dispatched on ``species_mode``.

    For backwards compatibility, if ``species_mode`` is not provided but
    ``species_emb_path`` is, we default to the legacy 'adapter' mode.
    """
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    val_df   = pd.read_csv(os.path.join(data_path, "val.csv"))
    test_df  = pd.read_csv(os.path.join(data_path, "test.csv"))

    # Legacy compat: if caller forgot to set species_mode but passed an emb path,
    # interpret that as 'adapter'.
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

    common_kwargs = dict(
        species_mode=species_mode,
        species_emb_map=species_emb_map,
        species_emb_dim=species_emb_dim,
        species_to_node_id=species_to_node_id,
    )

    train_dataset = MIC_Dataset(train_df, **common_kwargs)
    val_dataset   = MIC_Dataset(val_df,   **common_kwargs)
    test_dataset  = MIC_Dataset(test_df,  **common_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )
    return train_loader, val_loader, test_loader
