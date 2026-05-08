#!/usr/bin/env python3
"""
AMP MIC Regression Inference Script (with optional species side-channel support)

Auto-detects which species channel a checkpoint was trained with by inspecting
its ``state_dict`` keys:

    * ``species_adapter.*`` keys  -> legacy PubMedBERT-adapter pathway.
    * ``species_gnn.*``     keys  -> taxonomy-DAG GNN pathway.

Both can co-exist (``species_mode='both'``). Vanilla checkpoints (no species
key) are inferred without any side-channel.

Usage
-----
    # Vanilla / adapter checkpoint
    python infer.py \
        --model_path /path/to/checkpoint.pth \
        --test_csv   /NAS/luyq/AMP_datasets/splits1/test.csv \
        --output_dir /root/PLM_AMP_Regression/test_results/.../csv \
        --species_emb_path /NAS/luyq/AMP_datasets/species_embeddings.pkl \
        --batch_size 32 --device 0

    # GNN checkpoint (additionally requires --taxo_graph_path)
    python infer.py \
        --model_path /NAS/luyq/PLM_AMP_Regression/gnn_runs/esm2-150M/v1_splits1/...val_best.pth \
        --test_csv   /NAS/luyq/AMP_datasets/splits1/test.csv \
        --output_dir /root/PLM_AMP_Regression/test_results/tax_gnn/esm150m/splits1_test/csv \
        --taxo_graph_path /NAS/luyq/AMP_datasets/taxonomy_graph.pt \
        --batch_size 32 --device 0

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

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts')
sys.path.insert(0, SCRIPTS_DIR)

from plm_models import ESM2, load_tokenizer
from data_loader import seq2token


# ------------------------------- helpers ------------------------------------
def parse_model_info(model_path: str):
    """Parse PLM type / ESM size from checkpoint filename.

    Example: esm2-150M_lr1e-05_bs32_ep29_val_best.pth
    Returns: plm_type='esm2-150M', esm_size='150M', model_name='..._val_best'
    """
    basename   = os.path.basename(model_path)
    model_name = os.path.splitext(basename)[0]
    plm_type   = model_name.split('_')[0]
    esm_size   = plm_type.split('-')[-1]
    return plm_type, esm_size, model_name


def detect_species_config(state_dict: dict) -> dict:
    """Inspect a saved state_dict and figure out the species side-channel
    configuration. Returns a config dict that can be passed straight to
    ``ESM2(**cfg)``.
    """
    has_adapter = any(k.startswith("species_adapter.") for k in state_dict)
    has_gnn     = any(k.startswith("species_gnn.")     for k in state_dict)

    if not has_adapter and not has_gnn:
        return dict(species_mode="none")

    if has_adapter and has_gnn:
        species_mode = "both"
    elif has_adapter:
        species_mode = "adapter"
    else:
        species_mode = "gnn"

    cfg: dict = dict(species_mode=species_mode)

    # ---------- adapter sub-config ----------
    if has_adapter:
        # SpeciesAdapter layout (see PLM_head.py):
        #   net.0  Linear(in_dim     -> bottleneck)
        #   net.4  Linear(bottleneck -> out_dim)
        w0 = state_dict["species_adapter.net.0.weight"]
        w4 = state_dict["species_adapter.net.4.weight"]
        bottleneck, in_dim = w0.shape
        out_dim, _         = w4.shape
        cfg.update(
            species_in_dim     = int(in_dim),
            species_bottleneck = int(bottleneck),
            species_out_dim    = int(out_dim),
        )

    # ---------- GNN sub-config ----------
    if has_gnn:
        # Hidden / output projection dimensions.
        # input_proj.weight: [hidden, in_dim]
        # output_proj.weight: [out_dim, hidden]
        gnn_hidden  = int(state_dict["species_gnn.gnn.input_proj.weight"].shape[0])
        gnn_out_dim = int(state_dict["species_gnn.gnn.output_proj.weight"].shape[0])

        # Number of layers = max(layer index) + 1.
        layer_idxs = set()
        for k in state_dict.keys():
            if k.startswith("species_gnn.gnn.layers."):
                layer_idxs.add(int(k.split(".")[3]))
        gnn_layers = (max(layer_idxs) + 1) if layer_idxs else 1

        # GAT vs GCN: GAT layers register `att_src` / `att_dst` parameters.
        is_gat   = any(k.endswith("att_src") for k in state_dict
                       if k.startswith("species_gnn.gnn.layers."))
        gnn_type = "gat" if is_gat else "gcn"

        # GAT heads come from the att_src tensor shape [1, heads, hidden].
        gnn_heads = 4
        if is_gat:
            att_src = state_dict["species_gnn.gnn.layers.0.att_src"]
            gnn_heads = int(att_src.shape[1])

        # Hier vs leaf fusion: hier creates `absent_emb` and `hier_proj`.
        gnn_fusion = "hier" if "species_gnn.absent_emb" in state_dict else "leaf"

        cfg.update(
            gnn_hidden = gnn_hidden,
            gnn_out_dim= gnn_out_dim,
            gnn_layers = gnn_layers,
            gnn_type   = gnn_type,
            gnn_heads  = gnn_heads,
            gnn_fusion = gnn_fusion,
        )

    return cfg


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
    """Stack PubMedBERT embeddings for a batch of species names ([B, emb_dim])."""
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
                  species_emb_path: Optional[str] = None,
                  taxo_graph_path: Optional[str] = None):
    plm_type, esm_size, model_name = parse_model_info(model_path)
    print(f"[Model Info] PLM type: {plm_type} | ESM size: {esm_size}")
    print(f"[Model Name] {model_name}")

    if torch.cuda.is_available() and str(device_id) != "cpu":
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")
    print(f"[Device] {device}")

    print("[Model] Loading checkpoint...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    species_cfg = detect_species_config(state_dict)
    species_mode = species_cfg["species_mode"]

    if species_mode == "none":
        print("[Model] Vanilla checkpoint -- no species side-channel.")
    elif species_mode == "adapter":
        print(f"[Model] SpeciesAdapter only (in={species_cfg['species_in_dim']}, "
              f"bottleneck={species_cfg['species_bottleneck']}, "
              f"out={species_cfg['species_out_dim']}).")
        if species_emb_path is None:
            raise ValueError("Adapter checkpoint requires --species_emb_path.")
    elif species_mode == "gnn":
        print(f"[Model] Taxonomy GNN only (type={species_cfg['gnn_type']}, "
              f"layers={species_cfg['gnn_layers']}, hidden={species_cfg['gnn_hidden']}, "
              f"out={species_cfg['gnn_out_dim']}, fusion={species_cfg['gnn_fusion']}"
              + (f", heads={species_cfg['gnn_heads']}" if species_cfg['gnn_type'] == 'gat' else "")
              + ").")
        if taxo_graph_path is None:
            raise ValueError("GNN checkpoint requires --taxo_graph_path.")
    elif species_mode == "both":
        print(f"[Model] Adapter + GNN (mode='both').")
        if species_emb_path is None or taxo_graph_path is None:
            raise ValueError("'both' checkpoint needs --species_emb_path AND --taxo_graph_path.")

    df = pd.read_csv(test_csv)
    print(f"[Data] Loaded {len(df)} samples from {test_csv}")
    if species_mode != "none" and "Target_Species" not in df.columns:
        raise KeyError(
            f"species_mode={species_mode!r} but the test CSV has no `Target_Species` column."
        )

    print("[Tokenizer] Loading...")
    tokenizer = load_tokenizer(plm_type)

    model_kwargs = dict(
        plm_output='mean',
        head_type='3MLP',
        finetune_plm=True,
        esm_size=esm_size,
        **{k: v for k, v in species_cfg.items() if k != "species_mode"},
        species_mode=species_mode,
    )
    if species_mode in ("gnn", "both"):
        model_kwargs["taxo_graph_path"] = taxo_graph_path

    model = ESM2(**model_kwargs)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Model] Missing keys (non-critical, e.g. graph buffers): "
              f"{[m for m in missing if 'init_features' not in m and 'edge_index' not in m and 'ancestors_per_species' not in m][:5]}")
    if unexpected:
        print(f"[Model] Unexpected keys: {unexpected[:5]}")
    model = model.to(device)
    model.eval()
    print("[Model] Checkpoint loaded.")

    emb_map = None
    emb_dim = None
    if species_mode in ("adapter", "both"):
        emb_map = load_species_emb_map(species_emb_path)
        emb_dim = species_cfg["species_in_dim"]
        n_covered = sum(1 for n in df["Target_Species"].astype(str).unique() if n in emb_map)
        n_total   = df["Target_Species"].nunique()
        print(f"[Species/Adapter] {n_covered}/{n_total} species covered "
              f"(dim={emb_dim}); missing -> zero-vector.")

    if species_mode in ("gnn", "both"):
        # The species_gnn module already has the {name -> sp_idx} mapping
        # internally; we just need to feed Python lists of names.
        known = set(model.species_gnn.species_to_sp_idx.keys())
        unique_in_csv = df["Target_Species"].astype(str).unique()
        n_known = sum(1 for n in unique_in_csv if n in known)
        print(f"[Species/GNN] {n_known}/{len(unique_in_csv)} species present in taxonomy graph.")
        missing_names = [n for n in unique_in_csv if n not in known]
        if missing_names:
            raise KeyError(
                f"{len(missing_names)} species in test CSV are absent from the "
                f"taxonomy graph (first few: {missing_names[:5]}). Rebuild graph."
            )

    sequences = df['Sequence'].tolist()
    species_col = df['Target_Species'].astype(str).tolist() if species_mode != "none" else None
    all_preds = []

    with torch.no_grad():
        for start in tqdm(range(0, len(sequences), batch_size), desc="Inference"):
            batch_seqs = sequences[start: start + batch_size]
            batch_seqs_padded = [seq.ljust(max_length, 'X') for seq in batch_seqs]
            input_ids = seq2token(batch_seqs_padded, tokenizer, device)

            fwd_kwargs = {}
            if species_mode in ("adapter", "both"):
                batch_sp = species_col[start: start + batch_size]
                fwd_kwargs["species_emb"] = build_species_batch(batch_sp, emb_map, emb_dim, device)
            if species_mode in ("gnn", "both"):
                batch_sp = species_col[start: start + batch_size]
                fwd_kwargs["species_names"] = batch_sp

            outputs = model(input_ids, **fwd_kwargs)
            preds = outputs.squeeze(-1).cpu().numpy()
            all_preds.extend(preds.tolist())

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
                    "(auto-detects SpeciesAdapter / GNN checkpoints)."
    )
    parser.add_argument('--model_path',   type=str, required=True)
    parser.add_argument('--test_csv',     type=str, required=True)
    parser.add_argument('--output_dir',   type=str, required=True)
    parser.add_argument('--species_emb_path', type=str, default=None,
                        help='Pickle with PubMedBERT species embeddings; required for adapter / both checkpoints.')
    parser.add_argument('--taxo_graph_path',  type=str, default=None,
                        help='Path to taxonomy_graph.pt; required for gnn / both checkpoints.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device',     type=str, default='0',
                        help='CUDA device id (e.g. 0, 1) or "cpu".')
    parser.add_argument('--max_length', type=int, default=70)

    args = parser.parse_args()

    run_inference(
        model_path       = args.model_path,
        test_csv         = args.test_csv,
        output_dir       = args.output_dir,
        batch_size       = args.batch_size,
        device_id        = args.device,
        max_length       = args.max_length,
        species_emb_path = args.species_emb_path,
        taxo_graph_path  = args.taxo_graph_path,
    )


if __name__ == '__main__':
    main()
