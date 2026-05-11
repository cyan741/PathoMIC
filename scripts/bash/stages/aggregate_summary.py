#!/usr/bin/env python
"""Aggregate metrics across all stages into a single summary.csv.

Walks ``V2_ROOT`` and pulls each run's ``train_metrics.csv`` plus the bucketed
test MSE printed in ``logs/<job_id>.log``. Writes the joined table to
``V2_ROOT/summary.csv``.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

V2_ROOT_DEFAULT = "/NAS/luyq/PLM_AMP_Regression/gnn_runs_v2"


def parse_log(log_path: Path):
    """Pull bucketed test MSE numbers from train.py stdout (regex-based)."""
    if not log_path.exists():
        return {}
    txt = log_path.read_text(errors="ignore")
    out = {}
    m = re.search(r"BUCKETED TEST EVAL\s+\(overall MSE = ([\d.]+),\s*N=(\d+)", txt)
    if m:
        out["overall_test_mse_log"] = float(m.group(1))
        out["overall_test_n"]       = int(m.group(2))
    for m in re.finditer(r"\[([\d.]+),([\d.infe]+)\)\s+(\d+)\s+(\d+)\s+([\d.]+|n/a)", txt):
        lo, hi, n, _, mse = m.groups()
        try:
            mse_v = float(mse) if mse != "n/a" else float("nan")
        except ValueError:
            mse_v = float("nan")
        key = f"bucket_{lo}_{hi}".replace(".0", "").replace("inf", "Inf")
        out[key] = mse_v
        out[key + "_n"] = int(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=V2_ROOT_DEFAULT)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    out_path = Path(args.output) if args.output else root / "summary.csv"

    rows = []
    for stage_dir in sorted(root.iterdir()):
        if not stage_dir.is_dir() or not stage_dir.name.startswith("stage"):
            continue
        log_dir = stage_dir / "logs"
        for run_dir in sorted(stage_dir.iterdir()):
            if not run_dir.is_dir() or run_dir == log_dir:
                continue
            csvs = list(run_dir.glob("*.csv"))
            if not csvs:
                continue
            df = pd.read_csv(csvs[0])
            if df.empty:
                continue
            row = {
                "stage":       stage_dir.name,
                "variant_full": run_dir.name,
                "best_val_mse": float(df.best_val_loss.iloc[-1]),
                "test_mse":     float(df.final_test_loss.iloc[-1]),
                "n_epochs":     int(df.epoch.max()),
            }
            row.update(parse_log(log_dir / f"{run_dir.name}.log"))
            rows.append(row)

    if not rows:
        print(f"No runs found under {root}", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    df.sort_values(["stage", "test_mse"], inplace=True)
    df.to_csv(out_path, index=False)
    print(f"[wrote] {out_path}  ({len(df)} rows)")
    # Summary print
    print(df[["stage", "variant_full", "test_mse", "best_val_mse", "n_epochs"]].to_string(index=False))


if __name__ == "__main__":
    main()
