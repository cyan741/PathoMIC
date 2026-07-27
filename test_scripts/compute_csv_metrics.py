#!/usr/bin/env python3
"""Compute regression metrics for predict_MIC vs Median_MIC across CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TRUE_COL = "Median_MIC"
PRED_COL = "predict_MIC"
FIG_DATA_DIR = Path("/home/luyq/PLM_AMP_Regression/fig_data")


def _clean_pairs(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = (
        (~np.isnan(y_true))
        & (~np.isnan(y_pred))
        & (~np.isinf(y_true))
        & (~np.isinf(y_pred))
    )
    return y_true[mask], y_pred[mask]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true, y_pred = _clean_pairs(y_true, y_pred)
    n = len(y_true)
    nan = float("nan")
    if n < 2:
        return {
            "n_samples": n,
            "MAE": nan,
            "MSE": nan,
            "RMSE": nan,
            "Spearman_rho": nan,
            "Spearman_p": nan,
            "Kendall_tau": nan,
            "Kendall_p": nan,
            "R2": nan,
        }

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    rho, rho_p = spearmanr(y_true, y_pred)
    tau, tau_p = kendalltau(y_true, y_pred)

    return {
        "n_samples": n,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "Spearman_rho": float(rho),
        "Spearman_p": float(rho_p),
        "Kendall_tau": float(tau),
        "Kendall_p": float(tau_p),
        "R2": r2,
    }


def metrics_for_csv(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    for col in (TRUE_COL, PRED_COL):
        if col not in df.columns:
            raise ValueError(f"{csv_path.name}: missing required column '{col}'")

    row = compute_metrics(df[TRUE_COL].values, df[PRED_COL].values)
    row["source_file"] = csv_path.name
    return row


def default_output_path(input_dir: Path) -> Path:
    name = input_dir.name
    if name == "csv":
        name = input_dir.parent.name
    return FIG_DATA_DIR / f"{name}_csv_metrics.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute MAE, MSE, RMSE, Spearman rho, Kendall tau, and R2 "
            f"for {PRED_COL} vs {TRUE_COL} over CSV files."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "/NAS/luyq/PLM_AMP_Regression/test_results/raw_sequence/esm150m/splits2_test/csv"
        ),
        help="Directory containing input CSV files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern for CSV files within --input-dir.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: fig_data/<input_dir_name>_csv_metrics.csv).",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    csv_files = sorted(input_dir.glob(args.pattern))
    if not csv_files:
        raise SystemExit(f"No CSV files matched: {input_dir / args.pattern}")

    rows = []
    for csv_path in csv_files:
        try:
            rows.append(metrics_for_csv(csv_path))
            print(f"[ok] {csv_path.name}")
        except Exception as exc:
            print(f"[skip] {csv_path.name}: {exc}")

    if not rows:
        raise SystemExit("No metrics computed.")

    out_path = args.out or default_output_path(input_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metric_cols = [
        "n_samples",
        "MAE",
        "MSE",
        "RMSE",
        "Spearman_rho",
        "Spearman_p",
        "Kendall_tau",
        "Kendall_p",
        "R2",
    ]
    out_df = pd.DataFrame(rows)[["source_file", *metric_cols]]
    for col in metric_cols[1:]:
        out_df[col] = out_df[col].astype(float).round(4)
    out_df.to_csv(out_path, index=False, float_format="%.4f")

    print(f"\nWrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
