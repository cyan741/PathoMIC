import argparse
import json
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_TYPES = ["Gram-positive", "Gram-negative", "Fungi"]


def cumulative_species_split(
    df: pd.DataFrame,
    type_order: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split each Type by species frequency (descending), and keep species disjoint
    across train/val/test within that Type.
    """
    train_indices = []
    val_indices = []
    test_indices = []

    for type_name in type_order:
        type_df = df[df["Type"] == type_name]
        if type_df.empty:
            continue

        counts = type_df["Target_Species"].value_counts()
        total = int(counts.sum())
        train_threshold = total * train_ratio
        val_threshold = total * val_ratio

        train_species = set()
        val_species = set()
        test_species = set()

        cumulative = 0
        val_cumulative = 0
        stage = "train"

        for species, count in counts.items():
            if stage == "train":
                train_species.add(species)
                cumulative += int(count)
                if cumulative >= train_threshold:
                    stage = "val"
                continue

            if stage == "val":
                val_species.add(species)
                val_cumulative += int(count)
                if val_cumulative >= val_threshold:
                    stage = "test"
                continue

            test_species.add(species)

        train_indices.append(type_df[type_df["Target_Species"].isin(train_species)].index)
        val_indices.append(type_df[type_df["Target_Species"].isin(val_species)].index)
        test_indices.append(type_df[type_df["Target_Species"].isin(test_species)].index)

    train_df = df.loc[np.concatenate(train_indices)].copy() if train_indices else df.iloc[0:0].copy()
    val_df = df.loc[np.concatenate(val_indices)].copy() if val_indices else df.iloc[0:0].copy()
    test_df = df.loc[np.concatenate(test_indices)].copy() if test_indices else df.iloc[0:0].copy()

    return train_df, val_df, test_df


def filter_dataframe(
    df: pd.DataFrame,
    sequence_col: str,
    target_col: str,
    type_col: str,
    valid_types: List[str],
    max_seq_len: int,
) -> pd.DataFrame:
    """
    Filter the dataframe to ensure it has the required columns, valid types, and clean data.
     - sequence_col: ensure max sequence length and string type
     - target_col: ensure numeric type and no missing values
     - type_col: ensure value in this column is in valid_types list
     - valid_types: list of valid Type values to include
     - max_seq_len: maximum allowed sequence length (inclusive)
     - Returns a cleaned dataframe with only the valid types and required columns, and no missing values
    """
    required_cols = [sequence_col, target_col, type_col, "Target_Species"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    out = df.copy()
    out = out[out[type_col].isin(valid_types)].copy() # Filter to only valid types
    out[sequence_col] = out[sequence_col].astype(str)
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")  # coerce: non-numeric to NaN
    out = out.dropna(subset=[sequence_col, target_col, "Target_Species", type_col])
    out = out[out[sequence_col].str.len() <= max_seq_len].copy()
    out = out.reset_index(drop=True)
    return out


def _split_species_sets(df: pd.DataFrame, type_col: str = "Type") -> Dict[str, Dict[str, List[str]]]:
    """
    helper to build a dictionary of the unique species in each split, organized by Type.
        - Returns a dict of the form, eg: {"train": {"gram+": ["Enterococcus faecalis", ...], "gram-": [...], "fungi": [...]}, "val": {...}, "test": {...}}
    """
    split_map: Dict[str, Dict[str, List[str]]] = {"train": {}, "val": {}, "test": {}}
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        for t in sorted(split_df[type_col].unique().tolist()): 
            split_map[split_name][t] = sorted(split_df[split_df[type_col] == t]["Target_Species"].unique().tolist())
    return split_map


def make_split_bundle(
    input_csv: str,
    output_dir: str,
    sequence_col: str = "Sequence",
    target_col: str = "Median_MIC",
    type_col: str = "Type",
    valid_types: List[str] = None,
    max_seq_len: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    save_csv: bool = True,
    save_pickle: bool = True,
) -> Dict[str, object]:
    """Build reusable AMP MIC train/val/test splits, and save them in multiple formats for easy reuse."""
    if valid_types is None:
        valid_types = DEFAULT_TYPES

    os.makedirs(output_dir, exist_ok=True)

    raw_df = pd.read_csv(input_csv)
    df = filter_dataframe(
        raw_df,
        sequence_col=sequence_col,
        target_col=target_col,
        type_col=type_col,
        valid_types=valid_types,
        max_seq_len=max_seq_len,
    )

    train_df, val_df, test_df = cumulative_species_split(
        df,
        type_order=valid_types,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    merged = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)

    stats = {
        "total": int(len(merged)),
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
        "types": valid_types,
        "sequence_col": sequence_col,
        "target_col": target_col,
        "max_seq_len": int(max_seq_len),
    }

    species_sets = _split_species_sets(merged, type_col=type_col)

    if save_csv:
        train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
        merged.to_csv(os.path.join(output_dir, "all_with_split.csv"), index=False)

    if save_pickle:
        bundle = {
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "all_df": merged,
            "stats": stats,
            "species_sets": species_sets,
        }
        with open(os.path.join(output_dir, "split_bundle.pkl"), "wb") as f:
            pickle.dump(bundle, f)

        compact = {
            "stats": stats,
            "train_idx": train_df.index.to_list(),
            "val_idx": val_df.index.to_list(),
            "test_idx": test_df.index.to_list(),
            "species_sets": species_sets,
        }
        with open(os.path.join(output_dir, "split_index.pkl"), "wb") as f:
            pickle.dump(compact, f)

    with open(os.path.join(output_dir, "split_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "species_sets": species_sets}, f, ensure_ascii=False, indent=2)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "all_df": merged,
        "stats": stats,
        "species_sets": species_sets,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reusable AMP MIC train/val/test splits.")
    parser.add_argument("--input_csv", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--sequence_col", default="Sequence", type=str)
    parser.add_argument("--target_col", default="Median_MIC", type=str)
    parser.add_argument("--type_col", default="Type", type=str)
    parser.add_argument("--valid_types", default=",".join(DEFAULT_TYPES), type=str)
    parser.add_argument("--max_seq_len", default=32, type=int)
    parser.add_argument("--train_ratio", default=0.8, type=float)
    parser.add_argument("--val_ratio", default=0.1, type=float)
    parser.add_argument("--no_save_csv", action="store_true")
    parser.add_argument("--no_save_pickle", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    valid_types = [x.strip() for x in args.valid_types.split(",") if x.strip()]

    result = make_split_bundle(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        sequence_col=args.sequence_col,
        target_col=args.target_col,
        type_col=args.type_col,
        valid_types=valid_types,
        max_seq_len=args.max_seq_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        save_csv=not args.no_save_csv,
        save_pickle=not args.no_save_pickle,
    )

    stats = result["stats"]
    print(
        f"Split done. total={stats['total']} train={stats['train']} "
        f"val={stats['val']} test={stats['test']}"
    )


if __name__ == "__main__":
    main()
