#!/usr/bin/env python3
"""Plot top/bottom pathogen record counts from cleaned PathoMIC records."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import pandas as pd

INPUT_CSV = Path("/NAS/luyq/AMP_datasets/PathoMIC/csvs/cleaned_MIC_values.csv")
OUTPUT_PNG = Path(
    "/home/luyq/PLM_AMP_Regression/fig_data/pathogen_record_counts_cleaned_MIC.png"
)
OUTPUT_PDF = OUTPUT_PNG.with_suffix(".pdf")
COLORS = {
    "Gram-positive": "#4477AA",
    "Gram-negative": "#EE7733",
    "Fungi": "#228833",
}


def resolve_font() -> str:
    """Prefer Arial, with an explicit sans-serif fallback."""
    for family in ("Arial", "Liberation Sans", "DejaVu Sans"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
            return family
        except ValueError:
            continue
    return "sans-serif"


def main() -> None:
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    required = {"Target_Species", "Type"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # One CSV row is one MIC record.
    counts = (
        df.groupby(["Target_Species", "Type"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(
            ["count", "Target_Species"],
            ascending=[False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    unknown_types = set(counts["Type"]).difference(COLORS)
    if unknown_types:
        raise ValueError(f"Unexpected pathogen types: {sorted(unknown_types)}")
    if len(counts) <= 50:
        raise ValueError("At least 51 pathogens are required for an omitted middle.")

    shown = pd.concat([counts.head(25), counts.tail(25)], ignore_index=True)
    omitted = len(counts) - len(shown)

    left_x = list(range(25))
    gap_width = 5
    right_x = list(range(25 + gap_width, 50 + gap_width))
    x = left_x + right_x

    font_family = resolve_font()
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.linewidth": 2.5,
            "xtick.major.width": 2.2,
            "ytick.major.width": 2.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(38, 24))
    ax.bar(
        x,
        shown["count"],
        color=shown["Type"].map(COLORS),
        width=0.86,
        edgecolor="none",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        shown["Target_Species"],
        rotation=90,
        ha="center",
        va="top",
        fontsize=30,
        fontweight="bold",
        fontstyle="italic",
    )
    ax.tick_params(axis="x", length=0, pad=12)
    ax.tick_params(axis="y", labelsize=32, width=2.2, length=9)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.set_ylabel("number of records", fontsize=40, fontweight="bold", labelpad=24)
    ax.set_xlabel("Pathogen species", fontsize=40, fontweight="bold", labelpad=38)
    ax.set_xlim(-0.7, right_x[-1] + 0.7)
    ax.set_ylim(0, counts["count"].max() * 1.08)
    ax.yaxis.grid(True, color="#D0D0D0", linewidth=1.4, alpha=0.65)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    gap_center = (left_x[-1] + right_x[0]) / 2
    ax.text(
        gap_center,
        counts["count"].max() * 0.47,
        f"...\n{omitted} pathogens\nomitted",
        ha="center",
        va="center",
        fontsize=35,
        fontweight="bold",
        linespacing=1.2,
    )

    legend_handles = [
        Patch(facecolor=COLORS[name], edgecolor="none", label=name)
        for name in ("Gram-positive", "Gram-negative", "Fungi")
    ]
    legend = ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        fontsize=35,
        borderpad=0.5,
        labelspacing=0.4,
        handlelength=1.4,
        handletextpad=0.6,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_edgecolor("#B0B0B0")
    legend.get_frame().set_facecolor("white")

    fig.subplots_adjust(left=0.075, right=0.985, top=0.965, bottom=0.44)
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, facecolor="white")
    fig.savefig(OUTPUT_PDF, facecolor="white")
    plt.close(fig)

    print(f"Font: {font_family}")
    print(
        f"Pathogens: {len(counts)}; shown: 25 highest + 25 lowest; "
        f"omitted: {omitted}"
    )
    print(f"Saved PNG: {OUTPUT_PNG}")
    print(f"Saved PDF: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
