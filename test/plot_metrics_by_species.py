#!/usr/bin/env python3
"""
Per-species metrics horizontal bar chart for all test result CSVs.

Layout: 3 rows (RMSE / Spearman / Kendall τ), species on x-axis
        sorted by sample count (descending), coloured by Type.
        A horizontal dashed mean line is drawn on each subplot.

Species with n < 5 are excluded (too few for reliable metrics).
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

METRICS    = ["RMSE", "Spearman", "Kendall τ"]
MIN_COUNT  = 5          # species with fewer samples are skipped

TYPE_COLOR = {
    "Gram-negative": "#4C9BE8",
    "Gram-positive": "#F4845F",
    "Fungi":         "#6EBF8B",
    "Unknown":       "#AAAAAA",
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    if len(y_true) >= 2:
        rho, _ = spearmanr(y_true, y_pred)
        tau, _ = kendalltau(y_true, y_pred)
    else:
        rho = tau = np.nan
    return {"RMSE": rmse, "Spearman": rho, "Kendall τ": tau}


def plot_species(csv_path: str, out_dir: str):
    df = pd.read_csv(csv_path)

    # ── per-species stats ────────────────────────────────────────────────────
    rows = []
    for species, grp in df.groupby("Target_Species"):
        if len(grp) < MIN_COUNT:
            continue
        m = compute_metrics(grp["Median_MIC"].values, grp["predict_MIC"].values)
        stype = grp["Type"].iloc[0] if "Type" in grp.columns else "Unknown"
        rows.append({
            "species": species,
            "type":    stype,
            "n":       len(grp),
            **m,
        })

    if not rows:
        print(f"  [SKIP – no species with n>={MIN_COUNT}] {os.path.basename(csv_path)}")
        return None

    sdf = pd.DataFrame(rows).sort_values("RMSE", ascending=False).reset_index(drop=True)
    n_sp = len(sdf)

    # ── figure layout ────────────────────────────────────────────────────────
    fig_w = max(14, n_sp * 0.28)          # ~0.28 inch per species
    fig_h = 12
    fig, axes = plt.subplots(3, 1, figsize=(fig_w, fig_h),
                             sharex=True,
                             gridspec_kw={"hspace": 0.18})

    x      = np.arange(n_sp)
    colors = [TYPE_COLOR.get(t, TYPE_COLOR["Unknown"]) for t in sdf["type"]]

    for ax, metric in zip(axes, METRICS):
        vals = sdf[metric].values

        # bars
        bars = ax.bar(x, vals, width=0.75, color=colors,
                      edgecolor="white", linewidth=0.3)

        # mean line (ignore NaN)
        finite = vals[~np.isnan(vals)]
        if len(finite):
            mean_val = finite.mean()
            ax.axhline(mean_val, color="crimson", linewidth=1.2,
                       linestyle="--", zorder=5,
                       label=f"mean = {mean_val:.3f}")
            ax.legend(fontsize=8, loc="upper right")

        ax.set_ylabel(metric, fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)

        # y range
        if len(finite):
            lo, hi = np.nanmin(vals), np.nanmax(vals)
            pad = max((hi - lo) * 0.12, 0.02)
            ax.set_ylim(max(0, lo - pad), hi + pad * 2.5)

    # x-axis labels on the bottom subplot only (shared x)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(
        [f"{r.species}\n(n={r.n})" for r in sdf.itertuples()],
        rotation=90, fontsize=6.5, ha="center"
    )

    # ── type legend ──────────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, label=t)
                      for t, c in TYPE_COLOR.items()
                      if t != "Unknown"]
    axes[0].legend(handles=legend_handles + axes[0].get_legend_handles_labels()[0][:1],
                   labels=[t for t in TYPE_COLOR if t != "Unknown"] +
                           axes[0].get_legend_handles_labels()[1][:1],
                   fontsize=8, loc="upper right",
                   ncol=2, framealpha=0.8)

    # ── title ────────────────────────────────────────────────────────────────
    title = os.path.basename(csv_path).replace("_test_results.csv", "")
    fig.suptitle(
        f"{title}   (species n≥{MIN_COUNT}, sorted by RMSE↓)",
        fontsize=10, y=1.002
    )

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{title}.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    for test_dir in TEST_DIRS:
        csv_files = sorted(glob.glob(os.path.join(test_dir, "*_test_results.csv")))
        if not csv_files:
            print(f"[SKIP] No CSVs in {test_dir}")
            continue

        out_dir = os.path.join(test_dir, "metrics_by_species")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n[{os.path.basename(test_dir)}]  {len(csv_files)} files  →  {out_dir}")
        for csv_path in csv_files:
            out = plot_species(csv_path, out_dir)
            if out:
                print(f"  saved: {os.path.basename(out)}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
