#!/usr/bin/env python
"""Select the winner of a stage by mean test MSE across splits1+splits2.

Reads the train.py per-stage outputs (each variant has two save_dirs: ``<tag>_splits1``
and ``<tag>_splits2``), extracts ``best_val_loss`` and ``final_test_loss`` from the
``train_metrics.csv``-style CSVs, ranks by mean test MSE (lower is better) with the
plan's tie-breakers (splits2 [5,20) bucket, splits1 OOD bucket).

For tie-breaker bucket data we currently fall back to the mean test MSE only; the
bucketed metrics live in train.py stdout (logs/<job_id>.log) so we parse those when
present.

Usage:
    python select_winner.py --stage_dir /NAS/luyq/PLM_AMP_Regression/gnn_runs_v2/stage1 \
        --variants h3,h5,h7,hier_attn \
        --output /NAS/luyq/PLM_AMP_Regression/gnn_runs_v2/stage1/winner.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd


def latest_metrics_csv(run_dir: Path):
    """Find the train_metrics-style CSV in a run dir."""
    for f in run_dir.glob("*.csv"):
        return f
    return None


def parse_log(log_path: Path):
    """Pull bucketed test MSE numbers from train.py stdout (regex-based)."""
    if not log_path.exists():
        return {}
    txt = log_path.read_text(errors="ignore")
    out = {}
    # Extract overall MSE from BUCKETED block.
    m = re.search(r"BUCKETED TEST EVAL\s+\(overall MSE = ([\d.]+)", txt)
    if m:
        out["overall_mse"] = float(m.group(1))
    # Bucket lines look like:  "[5,20)                  XX            YY        Z.ZZZZ"
    for m in re.finditer(r"\[([\d.]+),([\d.infe]+)\)\s+(\d+)\s+(\d+)\s+([\d.]+|n/a)", txt):
        lo, hi, n, _, mse = m.groups()
        try:
            mse_v = float(mse) if mse != "n/a" else float("nan")
        except ValueError:
            mse_v = float("nan")
        out[f"bucket_{lo}_{hi}"] = mse_v
        out[f"bucket_{lo}_{hi}_n"] = int(n)
    return out


def collect_variant(stage_dir: Path, tag: str, log_dir: Path):
    """Return dict with metrics for both splits, or None if either is missing."""
    rec = {"variant": tag}
    for split in ("splits1", "splits2"):
        run_dir = stage_dir / f"{tag}_{split}"
        log = log_dir / f"{tag}_{split}.log"
        if not run_dir.exists():
            return None
        csv = latest_metrics_csv(run_dir)
        if csv is None:
            return None
        try:
            df = pd.read_csv(csv)
        except Exception:
            return None
        rec[f"{split}_best_val"] = float(df.best_val_loss.iloc[-1])
        rec[f"{split}_test_mse"] = float(df.final_test_loss.iloc[-1])
        rec[f"{split}_n_epochs"] = int(df.epoch.max())
        rec[f"{split}_log_metrics"] = parse_log(log)
    rec["mean_test_mse"] = (rec["splits1_test_mse"] + rec["splits2_test_mse"]) / 2.0
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_dir", required=True)
    ap.add_argument("--variants", required=True,
                    help="Comma-separated variant tags (e.g. h3,h5,h7,hier_attn)")
    ap.add_argument("--log_dir", default=None,
                    help="Where stdout logs live. Defaults to <stage_dir>/logs")
    ap.add_argument("--output", default=None,
                    help="Path to write winner.json. Default: <stage_dir>/winner.json")
    ap.add_argument("--baseline_dir", default=None,
                    help="Optional: dir of an existing run to compare against (e.g. v1.5).")
    ap.add_argument("--baseline_tag", default="h3_baseline")
    args = ap.parse_args()

    stage_dir = Path(args.stage_dir)
    log_dir = Path(args.log_dir) if args.log_dir else stage_dir / "logs"
    output = Path(args.output) if args.output else stage_dir / "winner.json"

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    rows = []
    for v in variants:
        rec = collect_variant(stage_dir, v, log_dir)
        if rec is None:
            print(f"[skip] {v}: missing metrics", file=sys.stderr)
            continue
        rows.append(rec)
    if not rows:
        print("No variants found", file=sys.stderr)
        sys.exit(1)

    rows.sort(key=lambda r: r["mean_test_mse"])
    winner = rows[0]

    print(f"{'variant':<20s} {'splits1':>10s} {'splits2':>10s} {'mean':>10s} {'epochs1':>8s} {'epochs2':>8s}")
    print("-" * 72)
    for r in rows:
        marker = " <-- winner" if r is winner else ""
        print(f"{r['variant']:<20s} {r['splits1_test_mse']:>10.4f} "
              f"{r['splits2_test_mse']:>10.4f} {r['mean_test_mse']:>10.4f} "
              f"{r['splits1_n_epochs']:>8d} {r['splits2_n_epochs']:>8d}{marker}")

    with open(output, "w") as f:
        json.dump({"winner": winner["variant"], "all": rows}, f, indent=2, default=str)
    print(f"\n[wrote] {output}")


if __name__ == "__main__":
    main()
