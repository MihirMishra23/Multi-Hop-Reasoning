#!/usr/bin/env python3
"""
Script to summarize lmlm results from the output/lmlm directory structure.
Usage: python print_multihop_table.py [lmlm_root] [--dataset DATASET] [--split SPLIT] [--format simple|markdown|latex]
"""

import json
import sys
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd


def find_results_files(lmlm_root: str) -> List[Path]:
    """Recursively find all results JSON files under lmlm_root."""
    root = Path(lmlm_root)
    if not root.exists():
        print(f"Error: Path '{lmlm_root}' does not exist")
        return []
    # Results files live under results_<date>/ subdirs
    return sorted(root.glob("*/*/results_*/*.json"))


def parse_path(path: Path) -> Dict[str, str]:
    """Extract dataset, model, and date from the file path."""
    # Expected: .../lmlm/{dataset}/{model}/results_{date}/{filename}.json
    parts = path.parts
    try:
        # Find the index of the results_* directory
        results_idx = next(i for i, p in enumerate(parts) if re.match(r"results_\d+", p))
        date = parts[results_idx][len("results_"):]
        model = parts[results_idx - 1]
        dataset = parts[results_idx - 2]
    except StopIteration:
        dataset = model = date = "unknown"
    return {"dataset": dataset, "model": model, "date": date}


def load_results(lmlm_root: str) -> List[Dict[str, Any]]:
    """Load all results from the lmlm directory structure."""
    files = find_results_files(lmlm_root)
    if not files:
        print(f"No results files found under '{lmlm_root}'")
        return []

    rows = []
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"Warning: could not load {f}: {e}")
            continue

        path_info = parse_path(f)
        metrics = data.get("metrics", {})
        meta = data.get("meta", {})

        # Infer split from filename if meta is missing it
        split = meta.get("split", "unknown")
        if split == "unknown":
            m = re.search(r"_(train[^_]*|dev|test)", f.stem)
            if m:
                split = m.group(1)

        rows.append({
            "Dataset": path_info["dataset"],
            "Model": path_info["model"],
            "Date": path_info["date"],
            "Split": split,
            "Count": metrics.get("count", 0),
            "EM": metrics.get("em", 0.0) * 100,
            "F1": metrics.get("f1", 0.0) * 100,
            "Precision": metrics.get("precision", 0.0) * 100,
            "Recall": metrics.get("recall", 0.0) * 100,
        })

    return rows


def build_table(rows: List[Dict[str, Any]],
                dataset_filter: Optional[str],
                split_filter: Optional[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if dataset_filter:
        df = df[df["Dataset"].str.lower() == dataset_filter.lower()]
    if split_filter:
        df = df[df["Split"].str.lower() == split_filter.lower()]

    # Format float columns
    for col in ("EM", "F1", "Precision", "Recall"):
        df[col] = df[col].map(lambda x: f"{x:.2f}")

    df = df.sort_values(["Dataset", "Split", "F1"], ascending=[True, True, False])
    return df.reset_index(drop=True)


def print_simple(df: pd.DataFrame):
    cols = ["Dataset", "Model", "Date", "Split", "Count", "EM", "F1", "Precision", "Recall"]
    display = df[[c for c in cols if c in df.columns]]
    print("\n" + "=" * 140)
    print("LMLM RESULTS SUMMARY")
    print("=" * 140 + "\n")
    # Print grouped by dataset
    for dataset, group in display.groupby("Dataset", sort=True):
        print(f"--- {dataset.upper()} ---")
        print(group.drop(columns=["Dataset"]).to_string(index=False))
        print()
    print("=" * 140)


def print_markdown(df: pd.DataFrame):
    cols = ["Dataset", "Model", "Date", "Split", "Count", "EM", "F1", "Precision", "Recall"]
    display = df[[c for c in cols if c in df.columns]]
    print("\n# LMLM Results\n")
    print(display.to_markdown(index=False))


def print_latex(df: pd.DataFrame):
    print("\n% LaTeX Table")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\begin{tabular}{l|l|c|c|c|c|c}")
    print("\\toprule")
    print("Dataset & Model & Split & Count & EM & F1 & Recall \\\\")
    print("\\midrule")
    for _, row in df.iterrows():
        model_short = row["Model"][:60] + ("..." if len(row["Model"]) > 60 else "")
        print(f"{row['Dataset']} & {model_short} & {row['Split']} & {row['Count']} & {row['EM']} & {row['F1']} & {row['Recall']} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{LMLM experimental results.}")
    print("\\label{tab:lmlm-results}")
    print("\\end{table}")


def main():
    parser = argparse.ArgumentParser(description="Summarize lmlm results.")
    parser.add_argument("lmlm_root", nargs="?",
                        default="/home/lz586/icl/Multi-Hop-Reasoning/output/lmlm",
                        help="Path to the lmlm output root directory")
    parser.add_argument("--dataset", "-d", default=None,
                        help="Filter by dataset (hotpotqa, musique, 2wiki, mquake)")
    parser.add_argument("--split", "-s", default=None,
                        help="Filter by split (dev, train, test, ...)")
    parser.add_argument("--format", "-f", default="simple",
                        choices=["simple", "markdown", "latex"],
                        help="Output format")
    args = parser.parse_args()

    print(f"Loading results from: {args.lmlm_root}")
    rows = load_results(args.lmlm_root)
    if not rows:
        return

    print(f"Loaded {len(rows)} result entries\n")

    df = build_table(rows, args.dataset, args.split)
    if df.empty:
        print("No results match the specified filters.")
        return

    if args.format == "latex":
        print_latex(df)
    elif args.format == "markdown":
        print_markdown(df)
    else:
        print_simple(df)


if __name__ == "__main__":
    main()
