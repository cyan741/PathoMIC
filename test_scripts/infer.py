#!/usr/bin/env python3
"""
AMP MIC Regression Inference Script (with optional species-adapter support)

Automatically detects whether the checkpoint was trained with a
`SpeciesAdapter` (i.e. ``model.use_species == True``) by looking for
``species_adapter.*`` keys in the saved state dict.  When present, a species
embedding pickle (list of ``{'pathogen','types','embedding','source'}`` dicts)
is loaded and each row's embedding is looked up by ``Target_Species``.

Usage
-----
    python infer.py \
        --model_path /NAS/luyq/PLM_AMP_Regression/ckp/ckp_species_text/splits1/esm2-8M_lr1e-05_bs32_ep22_val_best.pth \
        --test_csv   /NAS/luyq/AMP_datasets/splits1/test.csv \
        --output_dir /home/luyq/PLM_AMP_Regression/test/species_text/splits1_test \
        --species_emb_path /home/luyq/AMP_datasets/species_embeddings.pkl \
        --batch_size 32 \
        --device 0

Output
------
    {output_dir}/{model_name}_test_results.csv
    = original CSV columns + `predict_MIC` column (log10 scale).
"""

import os
import sys
import argparse
import pickle
from typing import Optional
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# --- Add scripts directory to sys.path so we can import project modules -----
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts')
sys.path.insert(0, SCRIPTS_DIR)

from plm_models import ESM2, load_tokenizer
from data_loader import seq2token


# ------------------------------- helpers ------------------------------------
def parse_model_info(model_path: str):
    """
    Parse PLM type and ESM size from checkpoint filename.

    Example: esm2-8M_lr1e-05_bs32_ep22_val_best.pth
    Returns: plm_type='esm2-8M', esm_size='8M', model_name='esm2-8M_lr1e-05_bs32_ep22_val_best'
    """
    basename   = os.path.basename(model_path)
    model_name = os.path.splitext(basename)[0]   # strip .pth
    plm_type   = model_name.split('_')[0]         # e.g. 'esm2-8M'
    esm_size   = plm_type.split('-')[-1]          # e.g. '8M'
    return plm_type, esm_size, model_name


def detect_species_config(state_dict: dict):
    """
    Peek into a saved state_dict and figure out whether the model was trained
    with a SpeciesAdapter.  Returns a config dict that can be passed directly
    to `ESM2(**cfg)`.
    """
    has_adapter = any(k.startswith("species_adapter.") for k in state_dict)
    if not has_adapter:
        return dict(use_species=False)

    # SpeciesAdapter layout (see PLM_head.py):
    #   net.0  Linear(in_dim  -> bottleneck)
    #   net.1  LayerNorm(bottleneck)
    #   net.2  GELU
    #   net.3  Dropout
    #   net.4  Linear(bottleneck -> out_dim)
    #   net.5  GELU
    w0 = state_dict["species_adapter.net.0.weight"]   # [bottleneck, in_dim]
    w4 = state_dict["species_adapter.net.4.weight"]   # [out_dim, bottleneck]
    bottleneck, in_dim = w0.shape
    out_dim, _          = w4.shape

    return dict(
        use_species=True,
        species_in_dim=int(in_dim),
        species_bottleneck=int(bottleneck),
        species_out_dim=int(out_dim),
    )


def load_species_emb_map(pkl_path: str) -> dict:
    """{pathogen_name -> FloatTensor[emb_dim]}"""
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


def build_species_batch(species_names, emb_map, emb_dim, device):
    """Stack embeddings for a batch of species names ([B, emb_dim] on device)."""
    zero = torch.zeros(emb_dim, dtype=torch.float32)
    tensors = [emb_map.get(str(n), zero) for n in species_names]
    return torch.stack(tensors, dim=0).to(device)


# ------------------------------- core ---------------------------------------
def run_inference(model_path: str,
                  test_csv: str,
                  output_dir: str,
                  batch_size: int = 32,
                  device_id: str = '0',
                  max_length: int = 70,
                  species_emb_path: Optional[str] = None):
    plm_type, esm_size, model_name = parse_model_info(model_path)
    print(f"[Model Info] PLM type: {plm_type} | ESM size: {esm_size}")
    print(f"[Model Name] {model_name}")

    # --- Device ----------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    print(f"[Device] {device}")

    # --- Load checkpoint first so we can inspect species config ----------
    print("[Model] Loading checkpoint...")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    species_cfg = detect_species_config(state_dict)
    if species_cfg["use_species"]:
        print(f"[Model] Detected SpeciesAdapter: in={species_cfg['species_in_dim']}, "
              f"bottleneck={species_cfg['species_bottleneck']}, "
              f"out={species_cfg['species_out_dim']}")
        if species_emb_path is None:
            raise ValueError(
                "Checkpoint uses SpeciesAdapter but --species_emb_path was not provided."
            )
    else:
        print("[Model] Vanilla model (no species adapter).")

    # --- Load test CSV ---------------------------------------------------
    df = pd.read_csv(test_csv)
    print(f"[Data] Loaded {len(df)} samples from {test_csv}")
    if species_cfg["use_species"] and "Target_Species" not in df.columns:
        raise KeyError(
            "Model expects species embedding but the test CSV has no `Target_Species` column."
        )

    # --- Load tokenizer --------------------------------------------------
    print("[Tokenizer] Loading...")
    tokenizer = load_tokenizer(plm_type)

    # --- Build model -----------------------------------------------------
    model = ESM2(
        plm_output='mean',
        head_type='3MLP',
        finetune_plm=True,
        esm_size=esm_size,
        **species_cfg,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Model] Missing keys (version mismatch, non-critical): {missing}")
    if unexpected:
        print(f"[Model] Unexpected keys: {unexpected}")
    model = model.to(device)
    model.eval()
    print("[Model] Checkpoint loaded.")

    # --- Load species embedding map (if needed) --------------------------
    emb_map = None
    emb_dim = None
    if species_cfg["use_species"]:
        emb_map = load_species_emb_map(species_emb_path)
        emb_dim = species_cfg["species_in_dim"]
        n_covered = sum(1 for n in df["Target_Species"].astype(str).unique() if n in emb_map)
        n_total   = df["Target_Species"].nunique()
        print(f"[Species] {n_covered}/{n_total} unique species covered by "
              f"{species_emb_path} (dim={emb_dim}); missing ones fall back to zero-vector.")

    # --- Inference -------------------------------------------------------
    sequences = df['Sequence'].tolist()
    species_col = df['Target_Species'].tolist() if species_cfg["use_species"] else None
    all_preds = []

    with torch.no_grad():
        for start in tqdm(range(0, len(sequences), batch_size), desc="Inference"):
            batch_seqs = sequences[start: start + batch_size]
            batch_seqs_padded = [seq.ljust(max_length, 'X') for seq in batch_seqs]
            input_ids = seq2token(batch_seqs_padded, tokenizer, device)

            if species_cfg["use_species"]:
                batch_sp = species_col[start: start + batch_size]
                species_emb = build_species_batch(batch_sp, emb_map, emb_dim, device)
                outputs = model(input_ids, species_emb=species_emb)   # [B, 1]
            else:
                outputs = model(input_ids)                            # [B, 1]

            preds = outputs.squeeze(-1).cpu().numpy()
            all_preds.extend(preds.tolist())

    # --- Save ------------------------------------------------------------
    df['predict_MIC'] = all_preds
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}_test_results.csv")
    df.to_csv(output_path, index=False)
    print(f"[Done] Saved {len(df)} rows -> {output_path}")
    return output_path


# ------------------------------- CLI ----------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a trained ESM2 MIC regression model "
                    "(auto-detects SpeciesAdapter checkpoints)."
    )
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to .pth checkpoint.')
    parser.add_argument('--test_csv', type=str, required=True,
                        help='Path to test CSV with `Sequence` (+ `Target_Species` for species models).')
    parser.add_argument('--output_dir', type=str,
                        default='/home/luyq/PLM_AMP_Regression/test',
                        help='Directory to save result CSV.')
    parser.add_argument('--species_emb_path', type=str,
                        default='/home/luyq/AMP_datasets/species_embeddings.pkl',
                        help='Pickle with pre-computed species embeddings (used only for species-adapter checkpoints).')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device id (e.g. 0, 1) or "cpu".')
    parser.add_argument('--max_length', type=int, default=70,
                        help='Sequence padding length (must match training setting).')

    args = parser.parse_args()

    run_inference(
        model_path=args.model_path,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device_id=args.device,
        max_length=args.max_length,
        species_emb_path=args.species_emb_path,
    )


if __name__ == '__main__':
    main()
