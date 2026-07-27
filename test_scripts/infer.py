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
        --batch_size 128 --device 0

    # Add this to either command to export the checkpoint-derived pathogen
    # vectors used by the fusion module (adapter or GNN, respectively):
        --export_species_repr /path/to/sp_adapter_output.npz

    # For GNN-RandomInit, reproduce the training-time random node features:
        --gnn_random_init 1 --gnn_random_init_seed 42 \
        --export_species_repr /path/to/gnn_random_init_output.npz

Output
------
    {output_dir}/{model_name}_test_results.csv
    = original CSV columns + `predict_MIC` column (log10 scale).

    Optional representation output (.npz): pathogen, embedding, types, source,
    and JSON metadata. Pass the same species_embeddings.pkl to all runs to keep
    pathogen order and plot labels identical across panels.
"""

import os
import re
import sys
import argparse
import pickle
import json
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


def parse_seed_from_path(model_path: str) -> Optional[int]:
    m = re.search(r"_sd(\d+)", os.path.basename(model_path))
    return int(m.group(1)) if m else None


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
        gnn_hidden  = int(state_dict["species_gnn.gnn.input_proj.weight"].shape[0])
        gnn_out_dim = int(state_dict["species_gnn.gnn.output_proj.weight"].shape[0])

        # Number of layers = max(layer index) + 1.
        layer_idxs = set()
        for k in state_dict.keys():
            if k.startswith("species_gnn.gnn.layers."):
                layer_idxs.add(int(k.split(".")[3]))
        gnn_layers = (max(layer_idxs) + 1) if layer_idxs else 1

        # GAT vs GCN: GAT layers register `att_src` / `att_dst` parameters.
        is_gat   = any(k.endswith("att_src") and k.startswith("species_gnn.gnn.layers.")
                       for k in state_dict)
        gnn_type = "gat" if is_gat else "gcn"

        gnn_heads = 4
        if is_gat:
            gnn_heads = int(state_dict["species_gnn.gnn.layers.0.att_src"].shape[1])

        # Fusion variant detection
        if "species_gnn.attn.in_proj_weight" in state_dict or \
           "species_gnn.attn.out_proj.weight" in state_dict:
            gnn_fusion = "hier_attn"
        elif "species_gnn.hier_proj.weight" in state_dict:
            gnn_fusion = "hier"
        else:
            gnn_fusion = "leaf"

        # Number of hier_levels (relevant for hier and hier_attn)
        gnn_hier_levels_n = None
        if "species_gnn.absent_emb" in state_dict:
            gnn_hier_levels_n = int(state_dict["species_gnn.absent_emb"].shape[0])

        # LoRA detection
        use_lora_init = "species_gnn.gnn.lora_A" in state_dict
        lora_rank = (int(state_dict["species_gnn.gnn.lora_A"].shape[1])
                     if use_lora_init else 16)

        # Residual/LayerNorm detection
        use_layernorm = any(k.startswith("species_gnn.gnn.norms.") for k in state_dict)
        # Residual is graph-only (no extra params), so we can't detect it from state_dict.
        # We default to False; user must explicitly set if needed (rare for inference).

        cfg.update(
            gnn_hidden = gnn_hidden,
            gnn_out_dim= gnn_out_dim,
            gnn_layers = gnn_layers,
            gnn_type   = gnn_type,
            gnn_heads  = gnn_heads,
            gnn_fusion = gnn_fusion,
            use_lora_init = use_lora_init,
            lora_rank  = lora_rank,
            gnn_layernorm = use_layernorm,
        )
        if gnn_hier_levels_n is not None:
            cfg["gnn_hier_levels_n"] = gnn_hier_levels_n

    # ---------- ESM<->GNN fusion strategy detection -------------------------
    # Anything besides simple concat creates `fusion.*` parameters.
    if any(k.startswith("fusion.") for k in state_dict):
        if "fusion.gate_proj.weight" in state_dict:
            cfg["fusion_strategy"] = "gated"
        elif "fusion.gamma_proj.weight" in state_dict:
            cfg["fusion_strategy"] = "film"
        elif "fusion.attn.in_proj_weight" in state_dict:
            cfg["fusion_strategy"] = "cross_attn"
        elif "fusion.proj_e.weight" in state_dict:
            cfg["fusion_strategy"] = "bilinear"
    else:
        cfg["fusion_strategy"] = "concat"

    return cfg


def apply_saved_training_config(detected: dict, saved: Optional[dict]) -> dict:
    """Prefer the exact architecture arguments stored by newer train.py runs."""
    if not saved:
        return detected
    cfg = dict(detected)
    mode = saved.get("species_mode")
    if mode is None and saved.get("use_species"):
        mode = "adapter"
    if mode is not None:
        cfg["species_mode"] = mode

    direct = (
        "species_out_dim", "species_bottleneck", "species_dropout",
        "gnn_hidden", "gnn_out_dim", "gnn_layers", "gnn_type", "gnn_heads",
        "gnn_dropout", "gnn_fusion", "use_lora_init", "lora_rank",
        "gnn_residual", "gnn_layernorm", "fusion_strategy", "species_inject",
        "prefix_pool", "prefix_depth", "prefix_kv_hidden", "lora_r",
        "lora_alpha", "lora_dropout",
    )
    for key in direct:
        if key in saved and saved[key] is not None:
            cfg[key] = saved[key]
    if saved.get("species_emb_dim") is not None:
        cfg["species_in_dim"] = int(saved["species_emb_dim"])
    if saved.get("gnn_hier_levels"):
        levels = saved["gnn_hier_levels"]
        cfg["gnn_hier_levels"] = tuple(levels.split(",")) if isinstance(levels, str) else tuple(levels)
        cfg.pop("gnn_hier_levels_n", None)
    if saved.get("lora_target"):
        targets = saved["lora_target"]
        cfg["lora_target"] = tuple(targets.split(",")) if isinstance(targets, str) else tuple(targets)
    for key in ("gnn_freeze_init", "freeze_esm", "use_lora"):
        if key in saved and saved[key] is not None:
            cfg[key] = bool(saved[key])
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


def load_species_records(pkl_path: str):
    """Load and validate the canonical pathogen order used for exports."""
    with open(pkl_path, "rb") as f:
        records = pickle.load(f)
    if not isinstance(records, (list, tuple)):
        raise TypeError("species embedding pickle must contain a list of records")
    names = [str(r["pathogen"]) for r in records]
    if len(names) != len(set(names)):
        raise ValueError("species embedding pickle contains duplicate pathogen names")
    return records


def export_species_representations(model, species_mode: str, output_path: str,
                                   model_path: str, device: torch.device,
                                   species_emb_path: Optional[str] = None,
                                   channel: Optional[str] = None,
                                   gnn_random_init: Optional[bool] = None,
                                   gnn_random_init_seed: Optional[int] = None):
    """Export one checkpoint-derived pathogen representation per pathogen.

    Adapter exports are the complete SpeciesAdapter outputs. GNN exports are
    the outputs of TaxonomySpeciesEncoder, i.e. the vectors actually consumed
    by the ESM--species fusion module. The export is independent of peptide
    sequences and is therefore computed once after loading the checkpoint.
    """
    available = []
    if species_mode in ("adapter", "both"):
        available.append("adapter")
    if species_mode in ("gnn", "both"):
        available.append("gnn")
    if channel is None:
        if len(available) != 1:
            raise ValueError("A 'both' checkpoint requires --export_species_repr_channel")
        channel = available[0]
    if channel not in available:
        raise ValueError(f"Cannot export channel={channel!r} from species_mode={species_mode!r}")

    records = load_species_records(species_emb_path) if species_emb_path else None
    if records is not None:
        names = [str(r["pathogen"]) for r in records]
        types = [str(r.get("types", "")) for r in records]
        sources = [str(r.get("source", "")) for r in records]
    elif channel == "gnn":
        by_index = sorted(model.species_gnn.species_to_sp_idx.items(), key=lambda kv: kv[1])
        names = [name for name, _ in by_index]
        types = [""] * len(names)
        sources = [""] * len(names)
    else:
        raise ValueError("Adapter export requires --species_emb_path")

    model.eval()
    with torch.no_grad():
        if channel == "adapter":
            x = []
            for r in records:
                emb = torch.as_tensor(r["embedding"], dtype=torch.float32)
                if emb.ndim != 1:
                    raise ValueError(f"Non-vector embedding for {r['pathogen']!r}: {tuple(emb.shape)}")
                x.append(emb)
            z = model.species_adapter(torch.stack(x).to(device))
            representation = "sp_adapter_output"
        else:
            known = model.species_gnn.species_to_sp_idx
            missing = [name for name in names if name not in known]
            if missing:
                raise KeyError(
                    f"{len(missing)} export pathogens are absent from the taxonomy graph "
                    f"(first few: {missing[:5]})"
                )
            z = model.species_gnn(names)
            representation = ("gnn_random_init_species_output" if gnn_random_init
                              else "gnn_species_output")

    embeddings = z.detach().float().cpu().numpy()
    if embeddings.ndim != 2 or embeddings.shape[0] != len(names):
        raise RuntimeError(f"Unexpected exported shape {embeddings.shape}; expected ({len(names)}, D)")
    if not np.isfinite(embeddings).all():
        raise ValueError("Exported representations contain NaN or infinity")

    metadata = {
        "representation": representation,
        "checkpoint": os.path.abspath(model_path),
        "species_mode": species_mode,
        "channel": channel,
        "n_pathogens": len(names),
        "dim": int(embeddings.shape[1]),
        "gnn_fusion": getattr(getattr(model, "species_gnn", None), "fusion", None),
        "gnn_random_init": bool(gnn_random_init) if channel == "gnn" else None,
        "gnn_random_init_seed": gnn_random_init_seed if channel == "gnn" else None,
    }
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        pathogen=np.asarray(names, dtype=str),
        embedding=embeddings.astype(np.float32, copy=False),
        types=np.asarray(types, dtype=str),
        source=np.asarray(sources, dtype=str),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    print(f"[Representation] Saved {embeddings.shape} {representation} -> {output_path}")
    return output_path


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
                  taxo_graph_path: Optional[str] = None,
                  gnn_random_init: Optional[bool] = None,
                  gnn_random_init_seed: Optional[int] = None,
                  export_species_repr: Optional[str] = None,
                  export_species_repr_channel: Optional[str] = None):
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
    saved_training_cfg = checkpoint.get("training_config")
    species_cfg = apply_saved_training_config(
        detect_species_config(state_dict), saved_training_cfg
    )
    species_mode = species_cfg["species_mode"]
    if saved_training_cfg:
        print("[Model] Restoring architecture from checkpoint training_config.")
        saved_plm = str(saved_training_cfg.get("plm", ""))
        if saved_plm.startswith("esm2-"):
            esm_size = saved_plm.split("-")[-1]

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

    # Resolve hier_levels by length when present (default ordering = first-N levels
    # closest to leaf; matches train.py default of species,genus,family for N=3).
    DEFAULT_LEVELS_FROM_LEAF = ("species", "genus", "family", "order",
                                 "class", "phylum", "kingdom", "domain")
    n_hier = species_cfg.pop("gnn_hier_levels_n", None)
    if n_hier is not None:
        species_cfg["gnn_hier_levels"] = DEFAULT_LEVELS_FROM_LEAF[:n_hier]

    model_kwargs = dict(
        plm_output=(saved_training_cfg or {}).get('plm_output', 'mean'),
        head_type=(saved_training_cfg or {}).get('head_type', '3MLP'),
        finetune_plm=(saved_training_cfg or {}).get('finetune_plm', True),
        esm_size=esm_size,
        **{k: v for k, v in species_cfg.items() if k != "species_mode"},
        species_mode=species_mode,
    )
    if species_mode in ("gnn", "both"):
        model_kwargs["taxo_graph_path"] = taxo_graph_path
        if gnn_random_init is None:
            if saved_training_cfg and saved_training_cfg.get("gnn_random_init") is not None:
                gnn_random_init = bool(saved_training_cfg["gnn_random_init"])
            else:
                gnn_random_init = "random_init" in model_path.lower()
        if gnn_random_init:
            if gnn_random_init_seed is None:
                if saved_training_cfg:
                    gnn_random_init_seed = saved_training_cfg.get("gnn_random_init_seed")
                    if gnn_random_init_seed is None:
                        gnn_random_init_seed = saved_training_cfg.get("seed")
                if gnn_random_init_seed is None:
                    gnn_random_init_seed = parse_seed_from_path(model_path)
            model_kwargs["gnn_random_init"] = True
            model_kwargs["gnn_random_init_seed"] = gnn_random_init_seed
            print(f"[Model] gnn_random_init=1 seed={gnn_random_init_seed}")

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

    if export_species_repr:
        export_species_representations(
            model=model,
            species_mode=species_mode,
            output_path=export_species_repr,
            model_path=model_path,
            device=device,
            species_emb_path=species_emb_path,
            channel=export_species_repr_channel,
            gnn_random_init=gnn_random_init,
            gnn_random_init_seed=gnn_random_init_seed,
        )

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
    parser.add_argument('--gnn_random_init', type=int, default=None, choices=[0, 1],
                        help='Use random GNN node features (auto-detected from '
                             '"random_init" in model_path when unset).')
    parser.add_argument('--gnn_random_init_seed', type=int, default=None,
                        help='Seed for random GNN features; defaults to _sd<N> in filename.')
    parser.add_argument('--export_species_repr', type=str, default=None,
                        help='Optional .npz path for checkpoint-derived pathogen representations.')
    parser.add_argument('--export_species_repr_channel', type=str, default=None,
                        choices=['adapter', 'gnn'],
                        help="Channel to export; required only for a 'both' checkpoint.")

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
        gnn_random_init  = (None if args.gnn_random_init is None
                            else bool(args.gnn_random_init)),
        gnn_random_init_seed = args.gnn_random_init_seed,
        export_species_repr = args.export_species_repr,
        export_species_repr_channel = args.export_species_repr_channel,
    )


if __name__ == '__main__':
    main()
