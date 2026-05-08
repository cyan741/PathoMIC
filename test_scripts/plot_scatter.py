#!/usr/bin/env python3
"""
Draw predicted-vs-true scatter plots for all test result CSVs.

For each folder in TEST_DIRS:
    - find all *_test_results.csv
    - plot predicted MIC vs true MIC with y=x reference line
    - title = filename without '_test_results'
    - save to {folder}/scatter_figs/{title}.png
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

TEST_DIRS = [
    "/home/luyq/PLM_AMP_Regression/test/raw_sequence/esm8m/splits1_test",
    "/home/luyq/PLM_AMP_Regression/test/raw_sequence/esm8m/splits2_test",
    "/home/luyq/PLM_AMP_Regression/test/raw_sequence/esm8m/external_test",
]


def plot_scatter(csv_path: str, out_dir: str):
    df = pd.read_csv(csv_path)
    y_true = df["Median_MIC"].values
    y_pred = df["predict_MIC"].values

    # --- metrics ---
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    fold_error = 10 ** mae  # average fold-difference on the original MIC scale
    # r2 = r2_score(y_true, y_pred)
    # r_pearson, p_value_pearson = pearsonr(y_true, y_pred)
    rho, p_value_rho = spearmanr(y_true, y_pred)
    tau, p_value_tau = kendalltau(y_true, y_pred)

    # --- title = filename without _test_results ---
    basename = os.path.basename(csv_path)                    # e.g. esm2-8M_lr1e-05_bs32_ep28_val_best_test_results.csv
    title = basename.replace("_test_results.csv", "")        # e.g. esm2-8M_lr1e-05_bs32_ep28_val_best

    # --- axis limits ---
    all_vals = np.concatenate([y_true, y_pred])
    vmin, vmax = all_vals.min(), all_vals.max()
    pad = (vmax - vmin) * 0.05
    lim = (vmin - pad, vmax + pad)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(y_true, y_pred, s=8, alpha=0.45, linewidths=0, color="#4C9BE8")

    # y = x reference line
    ax.plot(lim, lim, color="red", linestyle="--", linewidth=1.2, label="y = x")

    # metrics text box
    metrics_text = (
        f"RMSE = {rmse:.3f}  (log₁₀)\n"
        f"MAE  = {mae:.3f}  (~{fold_error:.2f}× fold)\n"
        # f"R²   = {r2:.3f}\n"
        f"Spearman = {rho:.3f}\n"
        f"Kendall = {tau:.3f}"
    )
    ax.text(0.04, 0.96, metrics_text,
            transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("True MIC (log₁₀)", fontsize=11)
    ax.set_ylabel("Predicted MIC (log₁₀)", fontsize=11)
    ax.set_title(title, fontsize=9, pad=6)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(fontsize=9, loc='lower right')

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{title}.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def main():
    for test_dir in TEST_DIRS:
        csv_files = sorted(glob.glob(os.path.join(test_dir, "csv", "*_test_results.csv")))
        if not csv_files:
            print(f"[SKIP] No result CSVs found in {test_dir}")
            continue

        out_dir = os.path.join(test_dir, "scatter_figs")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n[{os.path.basename(test_dir)}]  {len(csv_files)} files  ->  {out_dir}")
        for csv_path in csv_files:
            out_path = plot_scatter(csv_path, out_dir)
            print(f"  saved: {os.path.basename(out_path)}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
