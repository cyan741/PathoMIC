#!/usr/bin/env python3
"""
Per-type metrics bar chart (RMSE / Spearman / Kendall τ).

Layout  : 1 row × 3 subplots, each metric one subplot.
X-axis  : Gram-negative / Gram-positive / Fungi  (n= in label)
Y-axis  : zoomed to data range (does NOT force start at 0)
Style   : matches reference image
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mean_squared_error

TEST_DIRS = [
    "/home/luyq/PLM_AMP_Regression/test/splits1_test",
    "/home/luyq/PLM_AMP_Regression/test/splits2_test",
    "/home/luyq/PLM_AMP_Regression/test/external_test",
]

TYPES = ["Gram-negative", "Gram-positive", "Fungi"]

TYPE_COLORS = {
    "Gram-negative": "#4C9BE8",
    "Gram-positive": "#F4845F",
    "Fungi":         "#6EBF8B",
}

METRIC_YLABELS = {
    "RMSE":      "RMSE",
    "Spearman":  "Spearman",
    "Kendall τ": "Kendall τ",
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rho, _ = spearmanr(y_true, y_pred)
    tau, _ = kendalltau(y_true, y_pred)
    return {"RMSE": rmse, "Spearman": rho, "Kendall τ": tau}


def plot_metrics_by_type(csv_path: str, out_dir: str):
    df = pd.read_csv(csv_path)

    # ── per-type metrics ─────────────────────────────────────────────────────
    results = {}
    counts  = {}
    for t in TYPES:
        sub = df[df["Type"] == t]
        counts[t] = len(sub)
        if len(sub) >= 2:
            results[t] = compute_metrics(
                sub["Median_MIC"].values, sub["predict_MIC"].values
            )
        else:
            results[t] = {m: np.nan for m in METRIC_YLABELS}

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 4.0))
    fig.subplots_adjust(wspace=0.35)

    bar_w = 0.50
    x     = np.arange(len(TYPES))

    for ax, metric in zip(axes, METRIC_YLABELS):
        vals   = [results[t][metric] for t in TYPES]
        colors = [TYPE_COLORS[t] for t in TYPES]

        bars = ax.bar(x, vals, width=bar_w, color=colors,
                      edgecolor="white", linewidth=0.5)

        # value labels on bars
        for bar, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)

        # ── zoomed y-axis (no forced zero start) ────────────────────────────
        finite = [v for v in vals if not np.isnan(v)]
        if finite:
            lo, hi  = min(finite), max(finite)
            span    = hi - lo if hi != lo else 0.1
            pad_lo  = span * 0.6          # room below min bar
            pad_hi  = span * 1.2          # room above value labels
            ax.set_ylim(lo - pad_lo, hi + pad_hi)

        # x ticks
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{t}\n(n={counts[t]})" for t in TYPES],
            fontsize=9
        )

        ax.set_title(metric, fontsize=12, fontweight="bold", pad=7)
        ax.set_ylabel(METRIC_YLABELS[metric], fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8.5)

    # ── suptitle ─────────────────────────────────────────────────────────────
    title = os.path.basename(csv_path).replace("_test_results.csv", "")
    fig.suptitle(title, fontsize=10, y=1.02)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{title}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    for test_dir in TEST_DIRS:
        csv_files = sorted(glob.glob(os.path.join(test_dir, "*_test_results.csv")))
        if not csv_files:
            print(f"[SKIP] No CSVs in {test_dir}")
            continue

        out_dir = os.path.join(test_dir, "metrics_by_type")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n[{os.path.basename(test_dir)}]  {len(csv_files)} files  →  {out_dir}")
        for csv_path in csv_files:
            out_path = plot_metrics_by_type(csv_path, out_dir)
            print(f"  saved: {os.path.basename(out_path)}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
