#!/usr/bin/env python3
"""Summarize bucket MSE for adapter shuffle vs true-pairing baseline (splits2)."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

BUCKETS = ((0, 5), (5, 20), (20, 100), (100, float("inf")))


def bucket_mse_from_csv(csv_path: Path, train_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    sp_count = pd.read_csv(train_csv)["Target_Species"].astype(str).value_counts().to_dict()
    y = df["Median_MIC"].astype(float).values
    p = df["predict_MIC"].astype(float).values
    species = df["Target_Species"].astype(str).values
    rows = [{"train_count_bucket": "overall", "n_test_samples": len(df),
             "n_unique_species": df["Target_Species"].nunique(),
             "mse": float(np.mean((y - p) ** 2))}]
    for lo, hi in BUCKETS:
        mask = np.array([lo <= sp_count.get(s, 0) < hi for s in species])
        if mask.sum() == 0:
            rows.append({"train_count_bucket": f"[{lo},{hi})", "n_test_samples": 0,
                         "n_unique_species": 0, "mse": float("nan")})
            continue
        sub_y, sub_p = y[mask], p[mask]
        rows.append({
            "train_count_bucket": f"[{lo},{hi})",
            "n_test_samples": int(mask.sum()),
            "n_unique_species": len(set(species[mask])),
            "mse": float(np.mean((sub_y - sub_p) ** 2)),
        })
    out = pd.DataFrame(rows)
    out["source"] = csv_path.name
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffle_bucket_csv",
                    default="/NAS/luyq/PLM_AMP_Regression/test_results/species_text/shuffle/ckp/bucket_summary.csv")
    ap.add_argument("--baseline_test_csv",
                    default="/NAS/luyq/PLM_AMP_Regression/test_results/species_text/esm150m/splits2_test/csv/esm2-150M_lr0.0001_bs64_ep44_val_best_test_results.csv")
    ap.add_argument("--train_csv", default="/NAS/luyq/AMP_datasets/splits2/train.csv")
    ap.add_argument("--out",
                    default="/NAS/luyq/PLM_AMP_Regression/test_results/species_text/shuffle/bucket_comparison.csv")
    args = ap.parse_args()

    parts = []
    shuffle_path = Path(args.shuffle_bucket_csv)
    if shuffle_path.exists():
        sh = pd.read_csv(shuffle_path)
        sh["variant"] = "adapter_shuffle"
        parts.append(sh)
    else:
        print(f"[warn] shuffle bucket csv not found: {shuffle_path}")

    baseline_path = Path(args.baseline_test_csv)
    if baseline_path.exists():
        bl = bucket_mse_from_csv(baseline_path, Path(args.train_csv))
        bl["variant"] = "adapter_true (lr1e-4 bs64)"
        bl = bl.rename(columns={"source": "file"})
        parts.append(bl)

    if not parts:
        raise SystemExit("No inputs found.")

    out = pd.concat(parts, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print("\n=== Bucket MSE comparison (splits2 test) ===")
    for variant in out["variant"].unique():
        sub = out[out["variant"] == variant]
        overall = sub[sub["train_count_bucket"] == "overall"]["mse"].iloc[0]
        print(f"\n[{variant}] overall MSE = {overall:.4f}")
        for _, r in sub[sub["train_count_bucket"] != "overall"].iterrows():
            mse = r["mse"]
            mse_s = f"{mse:.4f}" if pd.notna(mse) else "n/a"
            print(f"  {r['train_count_bucket']:<12} n={int(r['n_test_samples']):5d}  mse={mse_s}")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
