#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_one(csv_path: Path, out_png: Path, title: str | None = None) -> None:
    df = pd.read_csv(csv_path)
    required = {"epoch", "train_mse", "val_mse"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

    fig = plt.figure(figsize=(8.5, 4.8), dpi=150)
    ax = fig.add_subplot(111)

    ax.plot(df["epoch"], df["train_mse"], label="train_mse", linewidth=1.6)
    ax.plot(df["epoch"], df["val_mse"], label="val_mse", linewidth=1.6)

    # Mark best val
    best_ep = int(df.loc[df["val_mse"].idxmin(), "epoch"])
    best_val = float(df["val_mse"].min())
    ax.scatter([best_ep], [best_val], s=25, zorder=3, label=f"best_val@ep{best_ep}={best_val:.4f}")

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (MSE)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Plot train/val MSE curves from metrics CSVs. "
            "Either pass --stage_dir (nested run folders) or one or more --csv (flat layout)."
        )
    )
    ap.add_argument(
        "--stage_dir",
        default=None,
        help="Parent dir containing subdirs per run, each with a metrics *.csv (legacy layout).",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Directory to write PNGs (created if missing).",
    )
    ap.add_argument(
        "--pattern",
        default="*.csv",
        help="With --stage_dir: glob inside each run dir (default: first match).",
    )
    ap.add_argument(
        "--csv",
        action="append",
        default=None,
        dest="csvs",
        metavar="PATH",
        help=(
            "Explicit metrics CSV path (repeatable). Example: HMAMP ckp splits1/splits2 "
            "HMAMP_lr1e-5_bs32.csv. Output: <out_dir>/<csv_stem>_loss.png"
        ),
    )
    ap.add_argument(
        "--title",
        default=None,
        help="Optional single title when exactly one --csv is given (ignored if multiple).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- flat: one or more explicit CSV paths --------------------------------
    if args.csvs:
        for i, csv_str in enumerate(args.csvs):
            csv_path = Path(csv_str).resolve()
            if not csv_path.is_file():
                raise FileNotFoundError(csv_path)
            stem = csv_path.stem
            parent_tag = csv_path.parent.name
            title = args.title if (args.title and len(args.csvs) == 1) else f"{parent_tag} / {stem}"
            out_png = out_dir / f"{parent_tag}_{stem}_loss.png"
            plot_one(csv_path, out_png, title=title)
            print(f"[ok] {csv_path} -> {out_png}")
        return

    # ----- legacy: stage_dir / <run> / *.csv -----------------------------------
    if not args.stage_dir:
        ap.error("Provide --stage_dir or at least one --csv.")
    stage_dir = Path(args.stage_dir)
    if not stage_dir.exists():
        raise FileNotFoundError(stage_dir)

    # stage_dir/<run_name>/<run_name>.csv
    run_dirs = [p for p in sorted(stage_dir.iterdir()) if p.is_dir() and p.name != "logs"]
    if not run_dirs:
        raise RuntimeError(f"No run dirs found under {stage_dir}")

    for run_dir in run_dirs:
        csvs = sorted(run_dir.glob(args.pattern))
        if not csvs:
            continue
        csv_path = csvs[0]
        run_name = run_dir.name
        out_png = out_dir / f"{run_name}_loss.png"
        plot_one(csv_path, out_png, title=run_name)
        print(f"[ok] {run_name}: {out_png}")


if __name__ == "__main__":
    main()

