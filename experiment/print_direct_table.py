#!/usr/bin/env python3
"""
Summarize direct-agent results.

Usage:
  python print_direct_table.py [root] [--dataset D] [--split S] [--format simple|markdown]

Scans {root}/{dataset}_{setting}/{model}/results JSON files.
"""

import json, argparse
from pathlib import Path
from typing import List, Dict
import pandas as pd

DIRECT_DEFAULT = "/home/lz586/icl/Multi-Hop-Reasoning/output/direct"


def load_results(root: str) -> List[Dict]:
    """Load all result files from {root}/{dataset}_{setting}/{model}/results."""
    rows = []
    for f in sorted(Path(root).glob("*/*/results")):
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            print(f"Warning: could not load {f}: {e}")
            continue
        meta = data.get("meta", {})
        metrics = data.get("metrics", {})
        dataset = meta.get("dataset") or f.parts[-3].split("_")[0]
        setting = meta.get("setting") or "_".join(f.parts[-3].split("_")[1:])
        model = meta.get("llm") or f.parts[-2]
        split = meta.get("split", "unknown")
        rows.append({
            "model": model,
            "dataset": dataset,
            "setting": setting,
            "split": split,
            "count": metrics.get("count", 0),
            "em": metrics.get("em", 0.0) * 100,
            "f1": metrics.get("f1", 0.0) * 100,
        })
    return rows


def build_wide(rows: List[Dict], dataset=None, split=None) -> pd.DataFrame:
    """Pivot to wide format: one row per model, columns = {dataset}_{setting}_{split}_em/f1."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if dataset:
        df = df[df["dataset"].str.lower() == dataset.lower()]
    if split:
        df = df[df["split"].str.lower() == split.lower()]
    if df.empty:
        return df

    df["col_em"] = df.apply(lambda r: f"{r['dataset']}_{r['setting']}_{r['split']}_em", axis=1)
    df["col_f1"] = df.apply(lambda r: f"{r['dataset']}_{r['setting']}_{r['split']}_f1", axis=1)

    em_wide = df.pivot_table(index="model", columns="col_em", values="em", aggfunc="first")
    f1_wide = df.pivot_table(index="model", columns="col_f1", values="f1", aggfunc="first")
    wide = em_wide.join(f1_wide, how="outer")
    wide.columns.name = None

    # Interleave em/f1 columns by dataset_setting_split, then avg columns
    base_keys = sorted({c.rsplit("_", 1)[0] for c in wide.columns})
    ordered = []
    for key in base_keys:
        for metric in ("em", "f1"):
            col = f"{key}_{metric}"
            if col in wide.columns:
                ordered.append(col)

    # Compute avg_em and avg_f1 over non-train columns
    em_cols = [c for c in ordered if c.endswith("_em") and "_train_" not in c]
    f1_cols = [c for c in ordered if c.endswith("_f1") and "_train_" not in c]
    if em_cols:
        wide["avg_em"] = wide[em_cols].mean(axis=1, skipna=True).round(2)
        ordered.append("avg_em")
    if f1_cols:
        wide["avg_f1"] = wide[f1_cols].mean(axis=1, skipna=True).round(2)
        ordered.append("avg_f1")

    # Sort by avg_em
    if "avg_em" in wide.columns:
        wide = wide.sort_values("avg_em", ascending=False, na_position="last")

    wide = wide[ordered].reset_index().rename(columns={"model": "Model"})

    counts = df.groupby("model")["count"].first().rename("Count")
    wide = wide.join(counts, on="Model")
    wide = wide[["Model", "Count"] + ordered]

    for col in ordered:
        wide[col] = wide[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")

    wide = wide.fillna("-").reset_index(drop=True)
    wide.insert(0, "Rank", range(1, len(wide) + 1))
    return wide


def print_table(wide: pd.DataFrame, fmt: str):
    if fmt == "markdown":
        print("\n# Results\n")
        print(wide.to_markdown(index=False))
    else:
        sep = "=" * max(80, 15 * len(wide.columns))
        print(f"\n{sep}\nRESULTS SUMMARY\n{sep}\n")
        print(wide.to_string(index=False))
        print(f"\n{sep}")


def main():
    parser = argparse.ArgumentParser(description="Summarize direct-agent results.")
    parser.add_argument("root", nargs="?", default=DIRECT_DEFAULT,
                        help=f"Path to direct output root (default: {DIRECT_DEFAULT})")
    parser.add_argument("--dataset", "-d", default=None, help="Filter by dataset")
    parser.add_argument("--split", "-s", default=None, help="Filter by split (dev/train/test)")
    parser.add_argument("--format", "-f", default="simple", choices=["simple", "markdown"])
    args = parser.parse_args()

    rows = load_results(args.root)
    print(f"Loaded {len(rows)} entries from {args.root}")

    wide = build_wide(rows, args.dataset, args.split)
    if wide.empty:
        print("No results match the specified filters.")
        return
    print()
    print_table(wide, args.format)


if __name__ == "__main__":
    main()
