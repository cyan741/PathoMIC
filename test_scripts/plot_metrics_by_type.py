#!/usr/bin/env python3
"""
Per-type metrics (RMSE / MAE / Spearman / Kendall τ).

By default: ONLY prints a per-file aligned table (columns = Gram-positive |
Gram-negative | Fungi), separated by vertical bars.
Plotting logic is kept (plot_metrics_by_type) but not invoked in main().
Toggle MAKE_PLOTS = True to re-enable the bar charts.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error

TEST_DIRS = [
    "/root/PLM_AMP_Regression/test_results/tax_gnn/esm150m/splits1_test",
    "/root/PLM_AMP_Regression/test_results/tax_gnn/esm150m/splits2_test",
    "/root/PLM_AMP_Regression/test_results/tax_gnn/esm150m/external_test",
]

# Column order used for the printed table (as requested)
TYPES = ["Gram-positive", "Gram-negative", "Fungi"]

# Metric rows in the printed table
METRIC_KEYS = ["RMSE", "MAE", "Spearman", "Kendall τ"]

# Flip to True to also save bar-chart figures.
MAKE_PLOTS = True

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
    mae = mean_absolute_error(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    tau, _ = kendalltau(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "Spearman": rho, "Kendall τ": tau}


def compute_per_type(df: pd.DataFrame) -> tuple[dict, dict]:
    """Return (results, counts) where results[type][metric] = value."""
    results, counts = {}, {}
    for t in TYPES:
        sub = df[df["Type"] == t]
        counts[t] = len(sub)
        if len(sub) >= 2:
            results[t] = compute_metrics(
                sub["Median_MIC"].values, sub["predict_MIC"].values
            )
        else:
            results[t] = {m: np.nan for m in METRIC_KEYS}
    return results, counts


def _fmt(v: float) -> str:
    return "   nan" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:7.4f}"


def print_metrics_table(title: str, results: dict, counts: dict) -> None:
    """Print an aligned table: one row per metric, columns = TYPES, sep = '|'."""
    headers = [f"{t} (n={counts[t]})" for t in TYPES]
    # Column width = max(header len, number-cell width), at least 14
    col_w = max(14, max(len(h) for h in headers))
    metric_col_w = max(len("Metric"), *(len(m) for m in METRIC_KEYS))

    sep = (
        "  "
        + "-" * metric_col_w
        + "-+-"
        + "-+-".join(["-" * col_w for _ in TYPES])
        + "-+"
    )

    print(f"\nFile: {title}")
    print(
        "  "
        + "Metric".ljust(metric_col_w)
        + " | "
        + " | ".join(h.center(col_w) for h in headers)
        + " |"
    )
    print(sep)
    for m in METRIC_KEYS:
        cells = [_fmt(results[t][m]).center(col_w) for t in TYPES]
        print("  " + m.ljust(metric_col_w) + " | " + " | ".join(cells) + " |")


def plot_metrics_by_type(results: dict, counts: dict, title: str, out_dir: str) -> str:
    """Original bar-chart rendering. Kept intact for later use."""
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 4.0))
    fig.subplots_adjust(wspace=0.35)

    bar_w = 0.50
    x     = np.arange(len(TYPES))

    # Plot only the three original metrics (RMSE / Spearman / Kendall τ)
    for ax, metric in zip(axes, METRIC_YLABELS):
        vals   = [results[t][metric] for t in TYPES]
        colors = [TYPE_COLORS[t] for t in TYPES]

        bars = ax.bar(x, vals, width=bar_w, color=colors,
                      edgecolor="white", linewidth=0.5)

        for bar, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)

        finite = [v for v in vals if not np.isnan(v)]
        if finite:
            lo, hi  = min(finite), max(finite)
            span    = hi - lo if hi != lo else 0.1
            pad_lo  = span * 0.6
            pad_hi  = span * 1.2
            ax.set_ylim(lo - pad_lo, hi + pad_hi)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{t}\n(n={counts[t]})" for t in TYPES],
            fontsize=9
        )
        ax.set_title(metric, fontsize=12, fontweight="bold", pad=7)
        ax.set_ylabel(METRIC_YLABELS[metric], fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8.5)

    fig.suptitle(title, fontsize=10, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{title}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    for test_dir in TEST_DIRS:
        csv_files = sorted(glob.glob(os.path.join(test_dir, "csv", "*_test_results.csv")))
        if not csv_files:
            print(f"[SKIP] No CSVs in {test_dir}")
            continue

        out_dir = os.path.join(test_dir, "metrics_by_type")
        if MAKE_PLOTS:
            os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'=' * 80}")
        print(f"[{os.path.basename(test_dir)}]  {len(csv_files)} files")
        print(f"{'=' * 80}")

        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            results, counts = compute_per_type(df)
            title = os.path.basename(csv_path).replace("_test_results.csv", "")

            print_metrics_table(title, results, counts)

            if MAKE_PLOTS:
                plot_metrics_by_type(results, counts, title, out_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
