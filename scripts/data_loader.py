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


class MIC_Dataset(Dataset):
    def __init__(self,
                 mic_df: pd.DataFrame,
                 max_length: int = 70,
                 species_emb_map: Optional[Dict[str, torch.Tensor]] = None,
                 species_emb_dim: int = 768):
        self.amp_seqs = mic_df.Sequence.tolist()
        self.mic_values = torch.tensor(mic_df.Median_MIC, dtype=torch.float32)  # tensor float
        self.max_length = max_length

        self.species_emb_map = species_emb_map
        self.species_emb_dim = species_emb_dim
        if species_emb_map is not None:
            if "Target_Species" not in mic_df.columns:
                raise KeyError(
                    "species_emb_map is given but the dataframe has no "
                    "`Target_Species` column."
                )
            self.species_names = mic_df.Target_Species.astype(str).tolist()
            self._zero_emb = torch.zeros(species_emb_dim, dtype=torch.float32)
        else:
            self.species_names = None
            self._zero_emb = None

    def __len__(self):
        return len(self.amp_seqs)

    def __getitem__(self, idx):
        sequence = self.amp_seqs[idx]
        # padding to max_length
        sequence = sequence.ljust(self.max_length, "X")
        mic_value = self.mic_values[idx]

        if self.species_emb_map is not None:
            sp_name = self.species_names[idx]
            sp_emb = self.species_emb_map.get(sp_name, self._zero_emb)
            return sequence, sp_emb, mic_value

        return sequence, mic_value

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
                species_emb_path: Optional[str] = None,
                species_emb_dim: int = 768):
    train_df, val_df, test_df = pd.read_csv(os.path.join(data_path, "train.csv")), \
                                 pd.read_csv(os.path.join(data_path, "val.csv")), \
                                 pd.read_csv(os.path.join(data_path, "test.csv"))

    species_emb_map = None
    if species_emb_path is not None:
        species_emb_map = load_species_embeddings(species_emb_path)
        print(f"[data_loader] loaded {len(species_emb_map)} species embeddings "
              f"(dim={species_emb_dim}) from {species_emb_path}")

    train_dataset = MIC_Dataset(train_df, species_emb_map=species_emb_map, species_emb_dim=species_emb_dim)
    val_dataset   = MIC_Dataset(val_df,   species_emb_map=species_emb_map, species_emb_dim=species_emb_dim)
    test_dataset  = MIC_Dataset(test_df,  species_emb_map=species_emb_map, species_emb_dim=species_emb_dim)
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
