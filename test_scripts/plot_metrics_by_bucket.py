#!/usr/bin/env python3
"""
Bucketed test MSE from saved inference CSVs (same logic as train.py ``bucketed_test_eval``).

For each ``*_test_results.csv`` under ``--csv_dir``, compute MSE from ``Median_MIC`` vs
``predict_MIC``, bucketed by the species's sample count in ``--train_csv``, print a
summary table, and save a bar chart next to the csv folder (``../metrics_by_bucket/``).

Usage
-----
    python plot_metrics_by_bucket.py \\
        --csv_dir /NAS/luyq/PLM_AMP_Regression/test_results/species_text/esm150m/splits2_test/csv \\
        --train_csv /NAS/luyq/AMP_datasets/splits2/train.csv
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_BUCKETS = ((0, 5), (5, 20), (20, 100), (100, float("inf")))


def load_species_train_counts(train_csv_path: str) -> dict[str, int]:
    train_df = pd.read_csv(train_csv_path)
    return train_df["Target_Species"].astype(str).value_counts().to_dict()


def bucketed_mse_from_csv(
    df: pd.DataFrame,
    sp_count: dict[str, int],
    buckets=DEFAULT_BUCKETS,
) -> dict:
    """Mirror ``train.py::bucketed_test_eval`` using precomputed CSV columns."""
    if "Median_MIC" not in df.columns or "predict_MIC" not in df.columns:
        raise KeyError("CSV must contain `Median_MIC` and `predict_MIC` columns.")
    if "Target_Species" not in df.columns:
        raise KeyError("CSV must contain `Target_Species` for bucket assignment.")

    y_true = df["Median_MIC"].astype(float).values
    y_pred = df["predict_MIC"].astype(float).values
    species = df["Target_Species"].astype(str).tolist()
    sq_errors = (y_pred - y_true) ** 2
    per_sample = list(zip(species, sq_errors.tolist()))

    overall_mse = float(np.mean(sq_errors)) if len(sq_errors) else float("nan")

    bucket_stats = []
    for lo, hi in buckets:
        sums = 0.0
        cnt = 0
        uniq_sp: set[str] = set()
        for name, s in per_sample:
            n_train = sp_count.get(name, 0)
            if lo <= n_train < hi:
                sums += s
                cnt += 1
                uniq_sp.add(name)
        mse = (sums / cnt) if cnt > 0 else float("nan")
        bucket_stats.append({
            "train_count_bucket": f"[{lo},{hi})",
            "n_test_samples": cnt,
            "n_unique_species": len(uniq_sp),
            "mse": mse,
        })

    return {"overall_mse": overall_mse, "n_samples": len(per_sample), "buckets": bucket_stats}


def print_bucket_report(title: str, report: dict) -> None:
    print("\n" + "=" * 72)
    print(f"BUCKETED TEST EVAL  {title}")
    print(f"  overall MSE = {report['overall_mse']:.4f}, N = {report['n_samples']}")
    print("=" * 72)
    print(f"{'train-count-bucket':<22}{'#test_samples':>16}{'#unique_sp':>14}{'mse':>14}")
    for b in report["buckets"]:
        mse_str = f"{b['mse']:.4f}" if not np.isnan(b["mse"]) else "n/a"
        print(
            f"{b['train_count_bucket']:<22}"
            f"{b['n_test_samples']:>16}"
            f"{b['n_unique_species']:>14}"
            f"{mse_str:>14}"
        )
    print("=" * 72)


def plot_bucket_bar(report: dict, title: str, out_path: str) -> None:
    buckets = report["buckets"]
    labels = [b["train_count_bucket"] for b in buckets]
    mses = [b["mse"] if not np.isnan(b["mse"]) else 0.0 for b in buckets]
    counts = [b["n_test_samples"] for b in buckets]
    colors = ["#4C9BE8", "#F4845F", "#6EBF8B", "#B388FF"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, mses, color=colors[: len(labels)], edgecolor="white", linewidth=0.8)

    ymax = max([m for m in mses if m > 0], default=0.1)
    for bar, mse, n in zip(bars, report["buckets"], counts):
        if np.isnan(mse["mse"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.01 * ymax,
                "n/a",
                ha="center",
                va="bottom",
                fontsize=9,
            )
            bar.set_height(0)
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02 * ymax,
            f"{mse['mse']:.3f}\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("MSE (log10 MIC)")
    ax.set_xlabel("Train-set species sample count bucket")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, ymax * 1.25 if ymax > 0 else 1.0)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Bucketed MSE stats + bar charts from inference CSVs."
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        default="/NAS/luyq/PLM_AMP_Regression/test_results/species_text/esm150m/splits2_test/csv",
        help="Directory containing *_test_results.csv files.",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="/NAS/luyq/AMP_datasets/splits2/train.csv",
        help="Training CSV used to count per-species sample frequency.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: sibling metrics_by_bucket/ next to csv_dir).",
    )
    args = parser.parse_args()

    csv_dir = os.path.abspath(args.csv_dir)
    out_dir = args.out_dir or os.path.join(os.path.dirname(csv_dir), "metrics_by_bucket")
    os.makedirs(out_dir, exist_ok=True)

    sp_count = load_species_train_counts(args.train_csv)
    print(f"[train] Loaded species counts from {args.train_csv} "
          f"({len(sp_count)} unique species)")

    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*_test_results.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No *_test_results.csv found under {csv_dir}")

    print(f"[csv] Found {len(csv_files)} files in {csv_dir}")
    print(f"[out] Plots -> {out_dir}\n")

    for csv_path in csv_files:
        basename = os.path.basename(csv_path)
        stem = basename.replace("_test_results.csv", "")
        title = stem

        df = pd.read_csv(csv_path)
        report = bucketed_mse_from_csv(df, sp_count)
        print_bucket_report(f"({basename})", report)

        plot_path = os.path.join(out_dir, f"{stem}.png")
        plot_bucket_bar(report, title, plot_path)


if __name__ == "__main__":
    main()
