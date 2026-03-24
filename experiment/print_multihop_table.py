#!/usr/bin/env python3
"""
Script to summarize lmlm results from the output/lmlm directory structure.
python /home/lz586/icl/Multi-Hop-Reasoning/experiment/print_multihop_table.py --date 0323
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
                split_filter: Optional[str],
                date_filter: Optional[str] = None) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if dataset_filter:
        df = df[df["Dataset"].str.lower() == dataset_filter.lower()]
    if split_filter:
        df = df[df["Split"].str.lower() == split_filter.lower()]
    if date_filter:
        df = df[df["Date"] == date_filter]

    df = df.sort_values(["Dataset", "Split", "F1"], ascending=[True, True, False])

    # Format float columns
    for col in ("EM", "F1", "Precision", "Recall"):
        df[col] = df[col].map(lambda x: f"{x:.2f}")
    return df.reset_index(drop=True)


def _build_wide(df: pd.DataFrame, tv1k_df: Optional[pd.DataFrame] = None,
                two_phase_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Pivot to one row per model: shared Split/Count, one EM col per dataset, Avg EM, ranked.
    tv1k_df: optional DataFrame of train-val1k results; added as extra columns, excluded from Avg.
    two_phase_df: optional DataFrame of two_phase results; added as extra columns, excluded from Avg."""
    slim = df[["Model", "Dataset", "Split", "Count", "EM"]].drop_duplicates(
        ["Model", "Dataset"], keep="first"
    )
    datasets = sorted(slim["Dataset"].unique())

    # Shared Split and Count (take from first dataset found per model)
    meta = slim.drop_duplicates("Model", keep="first")[["Model", "Split", "Count"]].set_index("Model")

    em_parts = []
    for ds in datasets:
        sub = (
            slim[slim["Dataset"] == ds][["Model", "EM"]]
            .rename(columns={"EM": ds})
            .set_index("Model")
        )
        em_parts.append(sub)

    wide = meta.join(pd.concat(em_parts, axis=1)).reset_index().fillna("-")

    # Compute numeric Avg over dataset EM columns (skip "-")
    em_cols = datasets
    wide["Avg"] = wide[em_cols].apply(
        lambda row: f"{sum(float(v) for v in row if v != '-') / max(sum(1 for v in row if v != '-'), 1):.2f}",
        axis=1,
    )

    # Append train-val1k columns after Avg (not included in Avg)
    if tv1k_df is not None and not tv1k_df.empty:
        tv1k_slim = tv1k_df[["Model", "Dataset", "EM"]].drop_duplicates(
            ["Model", "Dataset"], keep="first"
        )
        for ds in sorted(tv1k_slim["Dataset"].unique()):
            col_name = f"{ds}_train-val1k"
            sub = (
                tv1k_slim[tv1k_slim["Dataset"] == ds][["Model", "EM"]]
                .rename(columns={"EM": col_name})
                .set_index("Model")
            )
            wide = wide.set_index("Model").join(sub).reset_index().fillna("-")

    # Append two_phase columns after train-val1k (not included in Avg)
    if two_phase_df is not None and not two_phase_df.empty:
        tp_slim = two_phase_df[["Model", "Dataset", "EM"]].drop_duplicates(
            ["Model", "Dataset"], keep="first"
        )
        for ds in sorted(tp_slim["Dataset"].unique()):
            col_name = f"{ds}_2phase"
            sub = (
                tp_slim[tp_slim["Dataset"] == ds][["Model", "EM"]]
                .rename(columns={"EM": col_name})
                .set_index("Model")
            )
            wide = wide.set_index("Model").join(sub).reset_index().fillna("-")

    wide = wide.sort_values("Avg", ascending=False).reset_index(drop=True)
    wide.insert(0, "Rank", range(1, len(wide) + 1))
    return wide


def print_simple(df: pd.DataFrame, tv1k_df: Optional[pd.DataFrame] = None,
                 two_phase_df: Optional[pd.DataFrame] = None):
    wide = _build_wide(df, tv1k_df, two_phase_df)
    sep = "=" * (20 + 12 * (len(wide.columns) - 1))
    print("\n" + sep)
    print("LMLM RESULTS SUMMARY")
    print(sep + "\n")
    print(wide.to_string(index=False))
    print("\n" + sep)


def print_markdown(df: pd.DataFrame, tv1k_df: Optional[pd.DataFrame] = None,
                   two_phase_df: Optional[pd.DataFrame] = None):
    wide = _build_wide(df, tv1k_df, two_phase_df)
    print("\n# LMLM Results\n")
    print(wide.to_markdown(index=False))



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
    parser.add_argument("--two-phase-root", default="/home/lz586/icl/Multi-Hop-Reasoning/output/two_phase/",
                        help="Path to two_phase output root (defaults to sibling 'two_phase' dir)")
    parser.add_argument("--dataset", "-d", default=None,
                        help="Filter by dataset (hotpotqa, musique, 2wiki, mquake)")
    parser.add_argument("--split", "-s", default=None,
                        help="Filter by split (dev, train, test, ...)")
    parser.add_argument("--date", default=None,
                        help="Filter by date (e.g. 0204, 0320)")
    parser.add_argument("--format", "-f", default="simple",
                        choices=["simple", "markdown", "latex"],
                        help="Output format")
    args = parser.parse_args()

    print(f"Loading results from: {args.lmlm_root}")
    rows = load_results(args.lmlm_root)
    if not rows:
        return

    print(f"Loaded {len(rows)} result entries\n")

    df = build_table(rows, args.dataset, args.split, args.date)
    if df.empty:
        print("No results match the specified filters.")
        return

    # Always load train-val1k as extra columns (independent of --split filter)
    tv1k_df = build_table(rows, args.dataset, "train", args.date)
    if tv1k_df.empty:
        tv1k_df = None

    # Load two_phase results as extra columns
    two_phase_root = args.two_phase_root
    if two_phase_root is None:
        # Auto-detect: replace the last 'lmlm' component with 'two_phase'
        two_phase_root = str(Path(args.lmlm_root).parent / "two_phase")
    two_phase_rows = load_results(two_phase_root) if Path(two_phase_root).exists() else []
    two_phase_df = build_table(two_phase_rows, args.dataset, args.split, args.date) if two_phase_rows else None
    if two_phase_df is not None and two_phase_df.empty:
        two_phase_df = None
    if two_phase_df is not None:
        print(f"Loaded {len(two_phase_rows)} two_phase result entries from {two_phase_root}\n")

    if args.format == "latex":
        print_latex(df)
    elif args.format == "markdown":
        print_markdown(df, tv1k_df, two_phase_df)
    else:
        print_simple(df, tv1k_df, two_phase_df)


if __name__ == "__main__":
    main()
