#!/usr/bin/env python3
"""
Summarize multi-hop reasoning results.

Usage:
  python print_multihop_table.py [one_phase_root] [--two_phase_root PATH]
         [--dataset D] [--split S] [--date DATE] [--format simple|markdown]

Columns are named {dataset}_{split}_{date}_{phase}. Filters select which
combinations appear. avg_{phase} averages non-train columns for that phase.
"""

import json, re, argparse
from pathlib import Path
from typing import List, Dict
import pandas as pd

ONE_PHASE_DEFAULT = "/home/lz586/icl/Multi-Hop-Reasoning/output/lmlm"
TWO_PHASE_DEFAULT = "/home/lz586/icl/Multi-Hop-Reasoning/output/two_phase"


def load_results(root: str, phase: str) -> List[Dict]:
    """Load all result JSONs from {root}/{dataset}/{model}/results_{date}/*.json."""
    rows = []
    for f in sorted(Path(root).glob("*/*/results_*/*.json")):
        parts = f.parts
        try:
            ri = next(i for i, p in enumerate(parts) if p.startswith("results_"))
        except StopIteration:
            continue
        date, model, dataset = parts[ri][len("results_"):], parts[ri - 1], parts[ri - 2]
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            print(f"Warning: could not load {f}: {e}")
            continue
        split = data.get("meta", {}).get("split", "unknown")
        if split == "unknown":
            m = re.search(r"_(train[^_]*|dev|test)_", f.stem)
            if m:
                split = m.group(1)
        metrics = data.get("metrics", {})
        rows.append({
            "model": model, "dataset": dataset, "split": split,
            "date": date, "phase": phase,
            "count": metrics.get("count", 0),
            "em": metrics.get("em", 0.0) * 100,
        })
    return rows


def build_wide(rows: List[Dict], dataset=None, split=None, date=None) -> pd.DataFrame:
    """Pivot to wide format: one row per model, columns = {dataset}_{split}_{date}_{phase}."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if dataset: df = df[df["dataset"].str.lower() == dataset.lower()]
    if split:   df = df[df["split"].str.lower() == split.lower()]
    if date:    df = df[df["date"] == date]
    if df.empty:
        return df

    df["col"] = df.apply(
        lambda r: f"{r['dataset']}_{r['split']}_{r['phase']}", axis=1
    )
    wide = df.pivot_table(index="model", columns="col", values="em", aggfunc="first")
    wide.columns.name = None

    # Compute avg_{phase} (numeric) over non-train columns before formatting
    for phase in sorted(df["phase"].unique()):
        phase_cols = [c for c in wide.columns if c.endswith(f"_{phase}") and "_train_" not in c]
        if phase_cols:
            wide[f"avg_{phase}"] = wide[phase_cols].mean(axis=1, skipna=True).round(2)

    # Sort by avg_1phase, then avg_2phase
    sort_col = next((c for c in ["avg_1phase", "avg_2phase"] if c in wide.columns), None)
    if sort_col:
        wide = wide.sort_values(sort_col, ascending=False, na_position="last")

    # Reorder: group columns by phase with avg at end of each group
    ordered = []
    for phase in ["1phase", "2phase"]:
        ordered += sorted(c for c in wide.columns if c.endswith(f"_{phase}") and not c.startswith("avg_"))
        if f"avg_{phase}" in wide.columns:
            ordered.append(f"avg_{phase}")
    wide = wide[ordered].reset_index().rename(columns={"model": "Model"})

    # Join representative count per model
    counts = df.groupby("model")["count"].first().rename("Count")
    wide = wide.join(counts, on="Model")
    wide = wide[["Model", "Count"] + ordered]

    # Format floats
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
    parser = argparse.ArgumentParser(description="Summarize multi-hop reasoning results.")
    parser.add_argument("one_phase_root", nargs="?", default=ONE_PHASE_DEFAULT,
                        help="Path to 1phase output root (default: output/lmlm)")
    parser.add_argument("--two_phase_root", default=TWO_PHASE_DEFAULT,
                        help="Path to 2phase output root (default: output/two_phase)")
    parser.add_argument("--dataset", "-d", default=None, help="Filter by dataset")
    parser.add_argument("--split", "-s", default=None, help="Filter by split (dev/train/test)")
    parser.add_argument("--date", default=None, help="Filter by date tag (e.g. 0323, 0323_debug)")
    parser.add_argument("--format", "-f", default="simple", choices=["simple", "markdown"])
    args = parser.parse_args()

    rows = []
    p1, p2 = Path(args.one_phase_root), Path(args.two_phase_root)
    if p1.exists() and p1.resolve() != p2.resolve():
        r = load_results(str(p1), "1phase")
        print(f"Loaded {len(r)} 1phase entries from {p1}")
        rows += r
    if p2.exists():
        r = load_results(str(p2), "2phase")
        print(f"Loaded {len(r)} 2phase entries from {p2}")
        rows += r

    wide = build_wide(rows, args.dataset, args.split, args.date)
    if wide.empty:
        print("No results match the specified filters.")
        return
    print()
    print_table(wide, args.format)


if __name__ == "__main__":
    main()
