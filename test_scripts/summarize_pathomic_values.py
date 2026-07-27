#!/usr/bin/env python3
"""Summarize statistics for PathoMIC *values.csv files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_DIR = Path("/NAS/luyq/AMP_datasets/PathoMIC/csvs")
DEFAULT_OUTPUT_DIR = Path("/home/luyq/PLM_AMP_Regression/fig_data")


def sequence_length_distribution(lengths: pd.Series) -> pd.DataFrame:
    counts = lengths.value_counts().sort_index()
    return pd.DataFrame({"sequence_length": counts.index, "count": counts.values})


def summarize_file(csv_path: Path) -> tuple[dict, pd.DataFrame | None]:
    df = pd.read_csv(csv_path)

    if "Median_MIC" in df.columns:
        mic_col = "Median_MIC"
    elif "log_MIC_Value(uM)" in df.columns:
        mic_col = "log_MIC_Value(uM)"
    else:
        mic_col = None

    seq_lengths = df["Sequence"].astype(str).str.len()
    type_counts = df["Type"].astype(str).value_counts().sort_values(ascending=False)

    summary = {
        "file": csv_path.name,
        "n_records": len(df),
        "n_target_species": df["Target_Species"].nunique(),
        "n_unique_sequences": df["Sequence"].nunique(),
        "sequence_length_min": int(seq_lengths.min()),
        "sequence_length_max": int(seq_lengths.max()),
        "sequence_length_mean": round(float(seq_lengths.mean()), 4),
        "sequence_length_median": float(seq_lengths.median()),
        "type_distribution": type_counts.to_dict(),
    }

    if mic_col is not None:
        mic = pd.to_numeric(df[mic_col], errors="coerce")
        summary["mic_column"] = mic_col
        summary["median_mic_min"] = round(float(mic.min()), 4)
        summary["median_mic_max"] = round(float(mic.max()), 4)
        summary["median_mic_mean"] = round(float(mic.mean()), 4)
        summary["median_mic_median"] = round(float(mic.median()), 4)
    else:
        summary["mic_column"] = None
        summary["median_mic_min"] = None
        summary["median_mic_max"] = None
        summary["median_mic_mean"] = None
        summary["median_mic_median"] = None

    length_df = sequence_length_distribution(seq_lengths)
    return summary, length_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PathoMIC values CSV files.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--pattern", default="*values.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    csv_files = sorted(args.input_dir.glob(args.pattern))
    if not csv_files:
        raise SystemExit(f"No files matched: {args.input_dir / args.pattern}")

    summaries = []
    for csv_path in csv_files:
        summary, length_df = summarize_file(csv_path)
        summaries.append(summary)

        print(f"\n{'=' * 72}")
        print(f"File: {summary['file']}")
        print(f"{'=' * 72}")
        print(f"1. Total records:              {summary['n_records']:,}")
        print(f"2. Unique Target_Species:      {summary['n_target_species']:,}")
        print(f"3. Unique Sequence:            {summary['n_unique_sequences']:,}")
        print(
            "   Sequence length distribution:"
            f" min={summary['sequence_length_min']},"
            f" max={summary['sequence_length_max']},"
            f" mean={summary['sequence_length_mean']},"
            f" median={summary['sequence_length_median']}"
        )
        print("   Length histogram (length -> count):")
        for _, row in length_df.iterrows():
            print(f"     {int(row['sequence_length']):>3}: {int(row['count']):>6,}")

        print("4. Type distribution:")
        for type_name, count in summary["type_distribution"].items():
            pct = 100.0 * count / summary["n_records"]
            print(f"     {type_name:<20} {count:>7,}  ({pct:5.2f}%)")

        print("5. Median_MIC range:")
        if summary["mic_column"]:
            print(f"     column: {summary['mic_column']}")
            print(
                f"     min={summary['median_mic_min']},"
                f" max={summary['median_mic_max']},"
                f" mean={summary['median_mic_mean']},"
                f" median={summary['median_mic_median']}"
            )
        else:
            print("     (Median_MIC column not found)")

        length_out = args.output_dir / f"pathomic_{csv_path.stem}_seq_length_dist.tsv"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        length_df.to_csv(length_out, sep="\t", index=False)

    overview_rows = []
    for s in summaries:
        overview_rows.append(
            {
                "file": s["file"],
                "n_records": s["n_records"],
                "n_target_species": s["n_target_species"],
                "n_unique_sequences": s["n_unique_sequences"],
                "seq_len_min": s["sequence_length_min"],
                "seq_len_max": s["sequence_length_max"],
                "seq_len_mean": s["sequence_length_mean"],
                "seq_len_median": s["sequence_length_median"],
                "mic_column": s["mic_column"],
                "mic_min": s["median_mic_min"],
                "mic_max": s["median_mic_max"],
                "mic_mean": s["median_mic_mean"],
                "mic_median": s["median_mic_median"],
            }
        )
    overview_path = args.output_dir / "pathomic_values_summary.tsv"
    pd.DataFrame(overview_rows).to_csv(overview_path, sep="\t", index=False)

    type_rows = []
    for s in summaries:
        for type_name, count in s["type_distribution"].items():
            type_rows.append(
                {
                    "file": s["file"],
                    "type": type_name,
                    "count": count,
                    "pct": round(100.0 * count / s["n_records"], 4),
                }
            )
    type_path = args.output_dir / "pathomic_values_type_distribution.tsv"
    pd.DataFrame(type_rows).to_csv(type_path, sep="\t", index=False)

    print(f"\nSaved overview: {overview_path}")
    print(f"Saved type distribution: {type_path}")


if __name__ == "__main__":
    main()
