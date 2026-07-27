#!/usr/bin/env python3
"""Run bucketed test MSE analysis for a directory of prediction CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


DEFAULT_INPUT_DIR = Path(
    "/NAS/luyq/PLM_AMP_Regression/test_results/"
    "raw_sequence/esm150m/splits2_test/csv"
)
DEFAULT_TRAIN_CSV = Path("/NAS/luyq/AMP_datasets/splits2/train.csv")
DEFAULT_TAXONOMY_GRAPH = Path("/NAS/luyq/AMP_datasets/taxonomy_graph.pt")
DEFAULT_OUTPUT_DIR = Path("/home/luyq/PLM_AMP_Regression/fig_data")

BUCKET_LABELS = {
    "b0": "[0,5)",
    "b1": "[5,20)",
    "b2": "[20,100)",
    "b3": "[100,inf)",
}
REQUIRED_COLUMNS = {
    "Target_Species",
    "Median_MIC",
    "predict_MIC",
}


def assign_bucket(train_count: int) -> str:
    """Assign a training-sample count to a non-overlapping bucket."""
    if train_count < 5:
        return "b0"
    if train_count < 20:
        return "b1"
    if train_count < 100:
        return "b2"
    return "b3"


def load_graph(path: Path) -> dict[str, Any]:
    """Load a taxonomy graph on CPU, supporting old and new PyTorch."""
    try:
        graph = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        graph = torch.load(path, map_location="cpu")
    if not isinstance(graph, dict):
        raise TypeError(f"Expected a dict in taxonomy graph, got {type(graph)}")

    required = {
        "species_names",
        "ancestors_per_species",
        "levels",
        "idx2name",
        "idx2taxid",
    }
    missing = required - set(graph)
    if missing:
        raise KeyError(f"Taxonomy graph is missing keys: {sorted(missing)}")
    if "genus" not in graph["levels"]:
        raise KeyError("Taxonomy graph does not contain a genus level")
    return graph


def build_species_mapping(
    species: set[str],
    train_counts: dict[str, int],
    graph: dict[str, Any],
) -> pd.DataFrame:
    """Build bucket, genus and sibling metadata for input species."""
    levels = list(graph["levels"])
    genus_level = levels.index("genus")
    names = [str(name) for name in graph["species_names"]]
    ancestors = graph["ancestors_per_species"]

    graph_species: dict[str, dict[str, Any]] = {}
    genus_members: dict[int, set[str]] = {}
    for row_idx, species_name in enumerate(names):
        genus_node = int(ancestors[row_idx, genus_level])
        if genus_node >= 0:
            genus_members.setdefault(genus_node, set()).add(species_name)
            genus_name = str(graph["idx2name"][genus_node])
            genus_taxid = int(graph["idx2taxid"][genus_node])
        else:
            genus_name = ""
            genus_taxid = None
        graph_species[species_name] = {
            "genus_node": genus_node,
            "genus_name": genus_name,
            "genus_taxid": genus_taxid,
        }

    rows = []
    for species_name in sorted(species):
        train_count = int(train_counts.get(species_name, 0))
        bucket = assign_bucket(train_count)
        tax = graph_species.get(species_name)

        if tax is None or tax["genus_node"] < 0:
            siblings: set[str] = set()
            high_frequency_siblings: set[str] = set()
            sibling_group = "taxonomy_missing"
            genus_name = ""
            genus_taxid = None
            taxonomy_found = False
        else:
            members = genus_members[tax["genus_node"]]
            siblings = members - {species_name}
            high_frequency_siblings = {
                sibling
                for sibling in siblings
                if assign_bucket(int(train_counts.get(sibling, 0))) == "b3"
            }
            if not siblings:
                sibling_group = "no_sibling"
            elif high_frequency_siblings:
                sibling_group = "has_high_frequency_sibling"
            else:
                sibling_group = "no_high_frequency_sibling"
            genus_name = tax["genus_name"]
            genus_taxid = tax["genus_taxid"]
            taxonomy_found = True

        rows.append(
            {
                "Target_Species": species_name,
                "train_sample_count": train_count,
                "bucket": bucket,
                "bucket_range": BUCKET_LABELS[bucket],
                "taxonomy_found": taxonomy_found,
                "genus": genus_name,
                "genus_taxid": genus_taxid,
                "sibling_group": sibling_group if bucket == "b1" else "not_b1",
                "n_sibling_species": len(siblings),
                "n_b3_sibling_species": len(high_frequency_siblings),
                "sibling_species": ";".join(sorted(siblings)),
                "b3_sibling_species": ";".join(sorted(high_frequency_siblings)),
            }
        )
    return pd.DataFrame(rows)


def summarize_group(
    df: pd.DataFrame,
    mask: pd.Series,
    source_file: str,
    analysis_group: str,
    bucket: str,
    sibling_group: str = "",
) -> dict[str, Any]:
    """Calculate counts and MSE for one subset."""
    subset = df.loc[mask]
    y_true = pd.to_numeric(subset["Median_MIC"], errors="coerce").to_numpy()
    y_pred = pd.to_numeric(subset["predict_MIC"], errors="coerce").to_numpy()
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    mse = float(np.mean((y_true[valid] - y_pred[valid]) ** 2)) if valid.any() else np.nan

    return {
        "source_file": source_file,
        "analysis_group": analysis_group,
        "bucket": bucket,
        "bucket_range": BUCKET_LABELS.get(bucket, ""),
        "sibling_group": sibling_group,
        "n_test_samples": len(subset),
        "n_valid_pairs": int(valid.sum()),
        "n_pathogen_species": subset["Target_Species"].nunique(),
        "MSE": mse,
    }


def analyze_csv(
    csv_path: Path,
    mapping: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Return bucket and b1 sibling-group summaries for one prediction CSV."""
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")

    df["Target_Species"] = df["Target_Species"].astype(str)
    metadata = mapping[
        ["Target_Species", "bucket", "sibling_group"]
    ].drop_duplicates("Target_Species")
    df = df.merge(metadata, on="Target_Species", how="left", validate="many_to_one")
    if df["bucket"].isna().any():
        missing_species = sorted(df.loc[df["bucket"].isna(), "Target_Species"].unique())
        raise KeyError(f"species absent from mapping: {missing_species[:5]}")

    rows = []
    for bucket in ("b0", "b1", "b2", "b3"):
        rows.append(
            summarize_group(
                df,
                df["bucket"] == bucket,
                csv_path.name,
                analysis_group="bucket",
                bucket=bucket,
            )
        )

    b1 = df["bucket"] == "b1"
    sibling_masks = (
        ("no_sibling", df["sibling_group"] == "no_sibling"),
        (
            "has_sibling",
            df["sibling_group"].isin(
                ["has_high_frequency_sibling", "no_high_frequency_sibling"]
            ),
        ),
        (
            "has_high_frequency_sibling",
            df["sibling_group"] == "has_high_frequency_sibling",
        ),
        (
            "no_high_frequency_sibling",
            df["sibling_group"] == "no_high_frequency_sibling",
        ),
    )
    for sibling_group, sibling_mask in sibling_masks:
        rows.append(
            summarize_group(
                df,
                b1 & sibling_mask,
                csv_path.name,
                analysis_group="b1_sibling",
                bucket="b1",
                sibling_group=sibling_group,
            )
        )

    taxonomy_missing = b1 & (df["sibling_group"] == "taxonomy_missing")
    if taxonomy_missing.any():
        rows.append(
            summarize_group(
                df,
                taxonomy_missing,
                csv_path.name,
                analysis_group="b1_sibling",
                bucket="b1",
                sibling_group="taxonomy_missing",
            )
        )
    return rows


def default_prefix(input_dir: Path) -> str:
    """Derive a useful output prefix even when the directory is named csv."""
    return input_dir.parent.name if input_dir.name.lower() == "csv" else input_dir.name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bucket test species by training frequency and calculate per-group MSE."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument(
        "--taxonomy-graph", type=Path, default=DEFAULT_TAXONOMY_GRAPH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output filename prefix; defaults to the input directory name.",
    )
    args = parser.parse_args()

    for path, label in (
        (args.input_dir, "input directory"),
        (args.train_csv, "training CSV"),
        (args.taxonomy_graph, "taxonomy graph"),
    ):
        if not path.exists():
            raise SystemExit(f"{label} not found: {path}")

    csv_files = sorted(args.input_dir.glob(args.pattern))
    if not csv_files:
        raise SystemExit(f"No files matched {args.input_dir / args.pattern}")

    train_df = pd.read_csv(args.train_csv, usecols=["Target_Species"])
    train_counts = (
        train_df["Target_Species"].astype(str).value_counts().astype(int).to_dict()
    )

    input_species: set[str] = set()
    valid_csv_files = []
    for csv_path in csv_files:
        header = pd.read_csv(csv_path, nrows=0)
        missing = REQUIRED_COLUMNS - set(header.columns)
        if missing:
            print(f"[skip] {csv_path.name}: missing columns {sorted(missing)}")
            continue
        species_col = pd.read_csv(csv_path, usecols=["Target_Species"])
        input_species.update(species_col["Target_Species"].astype(str).unique())
        valid_csv_files.append(csv_path)
    if not valid_csv_files:
        raise SystemExit("No valid prediction CSV files were found.")

    graph = load_graph(args.taxonomy_graph)
    mapping = build_species_mapping(input_species, train_counts, graph)

    summary_rows = []
    for csv_path in valid_csv_files:
        summary_rows.extend(analyze_csv(csv_path, mapping))
        print(f"[ok] {csv_path.name}")

    prefix = args.output_prefix or default_prefix(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"{prefix}_bucketed_test.tsv"
    mapping_path = args.output_dir / f"{prefix}_species_bucket_mapping.tsv"

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_path, sep="\t", index=False, float_format="%.4f")
    mapping.to_csv(mapping_path, sep="\t", index=False)

    print(f"\nAnalyzed {len(valid_csv_files)} CSV files.")
    print(f"Summary: {summary_path}")
    print(f"Species mapping: {mapping_path}")


if __name__ == "__main__":
    main()
