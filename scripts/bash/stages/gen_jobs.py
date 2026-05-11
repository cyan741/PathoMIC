#!/usr/bin/env python
"""Generate `stage{N}_jobs.txt` for stages 2-5 by combining the previous
stage's winner.json with this stage's variant table.

Usage:
    python gen_jobs.py --stage 2 --output stage2_jobs.txt

Or for any stage explicitly:
    python gen_jobs.py --stage 4a --output stage4a_jobs.txt --winner stage3/winner.json

By default it auto-loads winner.json from the immediately preceding stage's
directory under V2_ROOT.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

V2_ROOT = "/NAS/luyq/PLM_AMP_Regression/gnn_runs_v2"
SCRIPTS = "/root/PLM_AMP_Regression/scripts"
DATA1 = "/NAS/luyq/AMP_datasets/splits1"
DATA2 = "/NAS/luyq/AMP_datasets/splits2"
TAXO  = "/NAS/luyq/AMP_datasets/taxonomy_graph.pt"
PY    = "/opt/conda/envs/esm-AMP/bin/python"

# ---------------------------------------------------------------------------
# Common training options reused across all stages.
# ---------------------------------------------------------------------------
COMMON = (
    " --finetune_plm True"
    " --plm esm2-150M"
    " --species_mode gnn"
    f" --taxo_graph_path {TAXO}"
    " --epochs 50 --use_scheduler --warmup_ratio 0.1 --min_lr_ratio 0.05"
    " --early_stopping --early_stop_patience 10 --es_min_epoch 20"
    " --device 0"
)


# ---------------------------------------------------------------------------
def winner_hp(winner: Dict) -> Dict:
    """Pull hyperparameters from a winner.json record. The winner.json schema
    is opinionated; we recompute most things from variant names below.
    """
    return winner


# ---------------------------------------------------------------------------
def render(tag: str, data_path: str, save_subdir: str, args: str) -> str:
    """One job line: <tag>|<command>."""
    metrics = f"{tag}.csv"
    save_dir = f"{V2_ROOT}/{save_subdir}/{tag}"
    return (
        f"{tag}|cd {SCRIPTS} && {PY} train.py"
        f" --data_path {data_path}"
        f"{COMMON}"
        f" --save_dir {save_dir} --metrics_name {metrics}"
        f" {args}"
    )


# ---------------------------------------------------------------------------
def stage2_jobs(prev_winner: Dict) -> List[str]:
    """Stage 2: learnable init (3 variants × 2 splits = 6 runs)."""
    # Inherit hier_levels + fusion from Stage 1 winner.
    fusion, levels = stage1_choices(prev_winner)
    hp = f' --gnn_fusion {fusion} --gnn_hier_levels "{levels}"'

    runs = [
        # (tag, extra_args)
        ("q2_unfrozen_lr1e5",  hp + " --gnn_freeze_init 0 --lr 1e-5 --batch_size 32"),
        ("q2_unfrozen_lr5e6",  hp + " --gnn_freeze_init 0 --lr 5e-6 --batch_size 32"),
        ("q2_lora_r16",        hp + " --gnn_freeze_init 1 --use_lora_init --lora_rank 16 --lr 1e-5 --batch_size 32"),
    ]
    out = []
    for tag, args in runs:
        out.append(render(f"{tag}_splits1", DATA1, "stage2", args))
        out.append(render(f"{tag}_splits2", DATA2, "stage2", args))
    return out


def stage1_choices(winner: Dict) -> Tuple[str, str]:
    """Map the winner variant name back to (fusion, hier_levels CSV)."""
    name = winner["winner"]
    if name == "h3":
        return "hier", "species,genus,family"
    if name == "h5":
        return "hier", "species,genus,family,order,class"
    if name == "h7":
        return "hier", "species,genus,family,order,class,phylum,kingdom"
    if name == "hier_attn":
        return "hier_attn", "domain,kingdom,phylum,class,order,family,genus,species"
    raise ValueError(f"Unknown Stage 1 winner: {name!r}")


# ---------------------------------------------------------------------------
def stage3_jobs(stage1_winner: Dict, stage2_winner: Dict) -> List[str]:
    """Stage 3: 5 fusion strategies × 2 splits = 10 runs."""
    fusion, levels = stage1_choices(stage1_winner)
    init_arg = stage2_init_arg(stage2_winner)
    hp_base = f' --gnn_fusion {fusion} --gnn_hier_levels "{levels}" {init_arg} --batch_size 32 --lr 1e-5'

    # Note: gated/film/cross_attn require gnn_out_dim == esm_hidden (=640 for 150M)
    runs = [
        ("f_concat_out64",   hp_base + " --gnn_out_dim 64  --fusion_strategy concat"),
        ("f_concat_out640",  hp_base + " --gnn_out_dim 640 --fusion_strategy concat"),
        ("f_gated",          hp_base + " --gnn_out_dim 640 --fusion_strategy gated"),
        ("f_film",           hp_base + " --gnn_out_dim 640 --fusion_strategy film"),
        ("f_xattn",          hp_base + " --gnn_out_dim 640 --fusion_strategy cross_attn"),
        ("f_bilinear",       hp_base + " --gnn_out_dim 64  --fusion_strategy bilinear"),
    ]
    # Stage 3 plan only calls for 5 fusion strategies, but we keep concat_out64
    # AND concat_out640 because the latter is an essential CONTROL for
    # "is the gain from wider GNN or from the new fusion?".
    out = []
    for tag, args in runs:
        out.append(render(f"{tag}_splits1", DATA1, "stage3", args))
        out.append(render(f"{tag}_splits2", DATA2, "stage3", args))
    return out


def stage2_init_arg(winner: Dict) -> str:
    """Map Stage 2 winner variant -> CLI flags."""
    name = winner["winner"]
    if name == "q2_unfrozen_lr1e5":
        return "--gnn_freeze_init 0"
    if name == "q2_unfrozen_lr5e6":
        return "--gnn_freeze_init 0"
    if name == "q2_lora_r16":
        return "--gnn_freeze_init 1 --use_lora_init --lora_rank 16"
    if name == "h3" or name.startswith("h"):
        # Stage 1 winner reused (frozen baseline)
        return "--gnn_freeze_init 1"
    return "--gnn_freeze_init 1"


# ---------------------------------------------------------------------------
def stage4a_jobs(stage3_winner: Dict) -> List[str]:
    """Stage 4a: lr × bs grid (3×3 = 9 configs × 2 splits = 18 runs)."""
    base = stage3_to_base(stage3_winner)
    runs = []
    for lr in (5e-6, 1e-5, 3e-5):
        for bs in (16, 32, 64):
            tag = f"4a_lr{lr:.0e}_bs{bs}"
            args = f"{base} --lr {lr} --batch_size {bs}"
            runs.append((tag, args))
    out = []
    for tag, args in runs:
        out.append(render(f"{tag}_splits1", DATA1, "stage4a", args))
        out.append(render(f"{tag}_splits2", DATA2, "stage4a", args))
    return out


def stage3_to_base(winner: Dict) -> str:
    """Convert Stage 3 winner into a CLI fragment (without lr/bs)."""
    # The variant name encodes fusion strategy and out_dim.
    name = winner["winner"]
    fmap = {
        "f_concat_out64":  ("concat", 64),
        "f_concat_out640": ("concat", 640),
        "f_gated":         ("gated",  640),
        "f_film":          ("film",   640),
        "f_xattn":         ("cross_attn", 640),
        "f_bilinear":      ("bilinear", 64),
    }
    fusion_strategy, out_dim = fmap.get(name, ("concat", 64))
    # Inherit hier+init from chained winners stored in winner.json
    levels = winner.get("hier_levels", "species,genus,family")
    fusion = winner.get("gnn_fusion", "hier")
    init_arg = winner.get("init_arg", "--gnn_freeze_init 1")
    return (
        f' --gnn_fusion {fusion} --gnn_hier_levels "{levels}" {init_arg}'
        f" --gnn_out_dim {out_dim} --fusion_strategy {fusion_strategy}"
    )


# ---------------------------------------------------------------------------
def stage4b_jobs(stage4a_winner: Dict) -> List[str]:
    """Stage 4b: gnn_lr_mult ∈ {5,10,20} (mult=1 already covered by Stage 4a winner).
    3 configs × 2 splits = 6 runs."""
    base = stage4a_to_base(stage4a_winner)
    runs = []
    for mult in (5, 10, 20):
        tag = f"4b_lrmult{mult}"
        args = f"{base} --gnn_lr_mult {mult}"
        runs.append((tag, args))
    out = []
    for tag, args in runs:
        out.append(render(f"{tag}_splits1", DATA1, "stage4b", args))
        out.append(render(f"{tag}_splits2", DATA2, "stage4b", args))
    return out


def stage4a_to_base(winner: Dict) -> str:
    """Inherit lr+bs+everything from Stage 4a winner."""
    return winner.get("base_args", "")  # Caller fills this out


# ---------------------------------------------------------------------------
def stage4c_jobs(stage4b_winner: Dict) -> List[str]:
    """Stage 4c: 6 architecture combos × 2 splits = 12 runs."""
    base = stage4b_winner.get("base_args", "")
    archs = [
        ("4c_L2_h128_oCur",  " --gnn_layers 2 --gnn_hidden 128"),
        ("4c_L3_h128",       " --gnn_layers 3 --gnn_hidden 128"),
        ("4c_L4_h128",       " --gnn_layers 4 --gnn_hidden 128"),
        ("4c_L2_h256_o2",    " --gnn_layers 2 --gnn_hidden 256"),
        ("4c_L3_h256_o2",    " --gnn_layers 3 --gnn_hidden 256"),
        ("4c_L2_h512_o4",    " --gnn_layers 2 --gnn_hidden 512"),
    ]
    out = []
    for tag, arch_args in archs:
        out.append(render(f"{tag}_splits1", DATA1, "stage4c", base + arch_args))
        out.append(render(f"{tag}_splits2", DATA2, "stage4c", base + arch_args))
    return out


# ---------------------------------------------------------------------------
def stage4d_jobs(stage4c_winner: Dict) -> List[str]:
    """Stage 4d: residual+LN + dropout sweep (4 configs × 2 splits = 8 runs)."""
    base = stage4c_winner.get("base_args", "")
    runs = [
        ("4d_baseline",         base + " --gnn_dropout 0.1"),
        ("4d_resLN_drop0",      base + " --gnn_residual --gnn_layernorm --gnn_dropout 0.0"),
        ("4d_resLN_drop01",     base + " --gnn_residual --gnn_layernorm --gnn_dropout 0.1"),
        ("4d_resLN_drop02",     base + " --gnn_residual --gnn_layernorm --gnn_dropout 0.2"),
    ]
    out = []
    for tag, args in runs:
        out.append(render(f"{tag}_splits1", DATA1, "stage4d", args))
        out.append(render(f"{tag}_splits2", DATA2, "stage4d", args))
    return out


# ---------------------------------------------------------------------------
def stage4e_jobs(stage4d_winner: Dict) -> List[str]:
    """Stage 4e: robust loss family (Huber, SmoothL1) × 2 splits = 4 runs."""
    base = stage4d_winner.get("base_args", "")
    runs = [
        ("4e_huber_d10",      base + " --loss_type huber --loss_huber_delta 1.0"),
        ("4e_smoothl1_b10",   base + " --loss_type smooth_l1 --loss_smooth_l1_beta 1.0"),
    ]
    out = []
    for tag, args in runs:
        out.append(render(f"{tag}_splits1", DATA1, "stage4e", args))
        out.append(render(f"{tag}_splits2", DATA2, "stage4e", args))
    return out


# ---------------------------------------------------------------------------
def stage4f_jobs(stage4e_winner: Dict) -> List[str]:
    """Stage 4f: imbalance-aware losses × 2 splits = 6 runs."""
    base = stage4e_winner.get("base_args", "")
    runs = [
        ("4f_lds_huber",  base + " --loss_type lds --loss_lds_base huber --loss_lds_sigma 2.0"),
        ("4f_bmc",        base + " --loss_type bmc --loss_bmc_noise 1.0"),
        ("4f_focal_r",    base + " --loss_type focal_r --loss_focal_gamma 2.0 --loss_focal_base mse"),
    ]
    out = []
    for tag, args in runs:
        out.append(render(f"{tag}_splits1", DATA1, "stage4f", args))
        out.append(render(f"{tag}_splits2", DATA2, "stage4f", args))
    return out


# ---------------------------------------------------------------------------
def stage4g_jobs(stage4f_winner: Dict) -> List[str]:
    """Stage 4g: GroupDRO (bucket grouping) × 2 splits = 2 runs."""
    base = stage4f_winner.get("base_args", "")
    runs = [
        ("4g_dro_bucket", base + " --loss_type group_dro --loss_dro_eta 0.01 --loss_dro_group_by bucket --loss_dro_base mse"),
    ]
    out = []
    for tag, args in runs:
        out.append(render(f"{tag}_splits1", DATA1, "stage4g", args))
        out.append(render(f"{tag}_splits2", DATA2, "stage4g", args))
    return out


# ---------------------------------------------------------------------------
def stage4h_jobs(stage4f_winner: Dict, stage4g_winner: Dict) -> List[str]:
    """Stage 4h: top-2 loss combo × 2 splits = 4 runs (2 combos)."""
    base = stage4f_winner.get("base_args", "")
    # Two combinations:
    #  - LDS-Huber + GroupDRO bucketing (re-weight + worst-group)
    #  - BMC + GroupDRO (BMC handles within-batch, DRO across groups)
    runs = [
        ("4h_lds_dro",  base + " --loss_type lds --loss_lds_base huber --loss_lds_sigma 2.0"),
        ("4h_bmc_dro",  base + " --loss_type bmc --loss_bmc_noise 1.0"),
    ]
    # Note: stacking is implemented inside the loss functions where applicable.
    # If we want a true "stack", a future TODO is to add `--loss_aux_type` to
    # mix two losses; for now we run them separately and let select_winner pick.
    out = []
    for tag, args in runs:
        out.append(render(f"{tag}_splits1", DATA1, "stage4h", args))
        out.append(render(f"{tag}_splits2", DATA2, "stage4h", args))
    return out


# ---------------------------------------------------------------------------
def stage5_jobs(final_args: str) -> List[str]:
    """Stage 5: 3 seeds × 2 splits + 3 controls × 2 splits = 12 runs."""
    base_long = final_args + " --epochs 80 --early_stop_patience 15 --es_min_epoch 25"
    runs: List[Tuple[str, str]] = []
    for seed in (42, 1337, 2025):
        runs.append((f"5_seed{seed}", base_long + f" --seed {seed}"))

    # Controls (use the same lr/bs/epochs schedule):
    runs.append(("5_ctrl_none",    base_long + " --species_mode none"))
    runs.append(("5_ctrl_adapter", base_long + " --species_mode adapter --species_emb_path /NAS/luyq/AMP_datasets/species_embeddings.pkl"))
    runs.append(("5_ctrl_both",    base_long + " --species_mode both --species_emb_path /NAS/luyq/AMP_datasets/species_embeddings.pkl"))

    out = []
    for tag, args in runs:
        out.append(render(f"{tag}_splits1", DATA1, "stage5", args))
        out.append(render(f"{tag}_splits2", DATA2, "stage5", args))
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    help="Stage id: 2, 3, 4a, 4b, 4c, 4d, 4e, 4f, 4g, 4h, or 5.")
    ap.add_argument("--output", required=True, help="Path to write the jobs.txt.")
    ap.add_argument("--winner_files", nargs="*", default=[],
                    help="winner.json files for prior stages, in order. Auto-resolves "
                         "from V2_ROOT/<prev_stage>/winner.json if omitted.")
    ap.add_argument("--final_args", default="",
                    help="(Stage 5 only) the full --gnn_* / --loss_* / --lr / --batch_size "
                         "args for the final winner config. See README.")
    args = ap.parse_args()

    def _load(p: str) -> Dict:
        with open(p) as f:
            return json.load(f)

    winners = [_load(p) for p in args.winner_files]

    if args.stage == "2":
        jobs = stage2_jobs(winners[0])
    elif args.stage == "3":
        jobs = stage3_jobs(winners[0], winners[1])
    elif args.stage == "4a":
        jobs = stage4a_jobs(winners[0])
    elif args.stage == "4b":
        jobs = stage4b_jobs(winners[0])
    elif args.stage == "4c":
        jobs = stage4c_jobs(winners[0])
    elif args.stage == "4d":
        jobs = stage4d_jobs(winners[0])
    elif args.stage == "4e":
        jobs = stage4e_jobs(winners[0])
    elif args.stage == "4f":
        jobs = stage4f_jobs(winners[0])
    elif args.stage == "4g":
        jobs = stage4g_jobs(winners[0])
    elif args.stage == "4h":
        jobs = stage4h_jobs(winners[0], winners[1])
    elif args.stage == "5":
        if not args.final_args:
            sys.exit("Stage 5 requires --final_args (the full final winner CLI fragment).")
        jobs = stage5_jobs(args.final_args)
    else:
        sys.exit(f"Unknown stage: {args.stage!r}")

    with open(args.output, "w") as f:
        f.write(f"# Auto-generated for stage {args.stage}\n")
        for line in jobs:
            f.write(line + "\n")
    print(f"[wrote] {args.output} ({len(jobs)} jobs)")


if __name__ == "__main__":
    main()
