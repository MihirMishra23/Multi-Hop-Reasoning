#!/usr/bin/env python3
"""
Summarize multi-hop reasoning results.

Usage:
  python print_multihop_table.py [one_phase_root] [--two_phase_root PATH]
         [--dataset D] [--split S] [--date DATE] [--format simple|markdown|long]
         [--use_contexts CTX] [--concat_all_db] [--use_train_params]

Columns are named {dataset}_{split}_{phase}. Settings (use_contexts, concat_all_db,
top_k, use_train_params) are shown as per-model metadata columns. When a model has
multiple results for the same dataset+split, the one with the highest EM is kept.

Use --format long to see one row per result file with all settings visible.
python /home/lz586/icl/Multi-Hop-Reasoning/experiment/print_multihop_table.py --date 0330 --format markdown --standard
"""

import json, re, argparse
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

ONE_PHASE_DEFAULT = "/home/lz586/icl/Multi-Hop-Reasoning/output/main_tables/lmlm"
TWO_PHASE_DEFAULT = "/home/lz586/icl/Multi-Hop-Reasoning/output/main_tables/two_phase"


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
        meta = data.get("meta", {})
        split = meta.get("split", "unknown")
        if split == "unknown":
            m = re.search(r"_(train[^_]*|dev|test)_", f.stem)
            if m:
                split = m.group(1)
        metrics = data.get("metrics", {})
        tp = meta.get("two_phase_params", {})
        inf = data.get("inference_params", {})
        rows.append({
            "model": model, "dataset": dataset, "split": split,
            "date": date, "phase": phase,
            "count": metrics.get("count", 0),
            "em": metrics.get("em", 0.0) * 100,
            "f1": metrics.get("f1", 0.0) * 100,
            "use_contexts": tp.get("use_contexts"),
            "concat_all_db": tp.get("concat_all_db"),
            "top_k": tp.get("top_k"),
            "use_train_params": inf.get("use_train_params"),
        })
    return rows


def _fmt_bool(x):
    return "Y" if x is True else ("N" if x is False else "-")


def build_wide(
    rows: List[Dict],
    dataset=None,
    split=None,
    date=None,
    use_contexts: Optional[str] = None,
    concat_all_db: Optional[bool] = None,
    use_train_params: Optional[bool] = None,
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    """Pivot to wide format: one row per (model, settings) combo, columns = {dataset}_{split}_{phase}.
    Each unique combination of settings gets its own row."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if dataset:          df = df[df["dataset"].str.lower() == dataset.lower()]
    if split:            df = df[df["split"].str.lower() == split.lower()]
    if date:             df = df[df["date"] == date]
    if use_contexts is not None:
        df = df[df["use_contexts"].str.lower() == use_contexts.lower()]
    if concat_all_db is not None:
        df = df[df["concat_all_db"] == concat_all_db]
    if use_train_params is not None:
        df = df[df["use_train_params"] == use_train_params]
    if top_k is not None:
        df = df[df["top_k"] == top_k]
    if df.empty:
        return df

    # Build a settings key to distinguish rows within the same model
    def _settings_key(r):
        parts = []
        if pd.notna(r.get("use_contexts")):
            parts.append(str(r["use_contexts"]))
        parts.append("cdb" if r.get("concat_all_db") is True else "no_cdb")
        if pd.notna(r.get("top_k")):
            parts.append(f"k{int(r['top_k'])}")
        parts.append("tp" if r.get("use_train_params") is True else "greedy")
        return "|".join(parts)

    df["settings_key"] = df.apply(_settings_key, axis=1)
    df["row_key"] = df["model"] + "  [" + df["settings_key"] + "]"
    df["col"] = df.apply(lambda r: f"{r['dataset']}_{r['split']}_{r['phase']}", axis=1)

    # Drop exact duplicates (same model+settings+col), keep best EM
    df = df.sort_values("em", ascending=False).drop_duplicates(subset=["row_key", "col"])

    wide = df.pivot_table(index="row_key", columns="col", values="em", aggfunc="max")
    wide.columns.name = None

    # Compute avg_{phase} over non-train columns
    for phase in sorted(df["phase"].unique()):
        phase_cols = [c for c in wide.columns if c.endswith(f"_{phase}") and "_train" not in c]
        if phase_cols:
            wide[f"avg_{phase}"] = wide[phase_cols].mean(axis=1, skipna=True).round(2)

    # Reorder: group columns by phase with avg at end of each group
    ordered = []
    for phase in ["1phase", "2phase"]:
        ordered += sorted(c for c in wide.columns if c.endswith(f"_{phase}") and not c.startswith("avg_"))
        if f"avg_{phase}" in wide.columns:
            ordered.append(f"avg_{phase}")
    wide = wide[ordered].reset_index().rename(columns={"row_key": "Model"})

    # Join per-row settings (now guaranteed unique per row)
    agg = df.groupby("row_key").first()[["model", "count", "use_contexts", "concat_all_db", "use_train_params", "top_k"]]
    wide = wide.join(agg, on="Model")
    wide = wide.rename(columns={"model": "ModelName", "count": "Count"})
    wide["UseCtx"]      = wide["use_contexts"].fillna("-")
    wide["ConcatDB"]    = wide["concat_all_db"].map(_fmt_bool)
    wide["TrainParams"] = wide["use_train_params"].map(_fmt_bool)
    wide["TopK"]        = wide["top_k"].apply(lambda x: str(int(x)) if pd.notna(x) else "-")

    # Sort by settings combo first, then by avg EM descending within each group
    avg_col = next((c for c in ["avg_1phase", "avg_2phase"] if c in wide.columns), None)
    sort_keys = ["UseCtx", "ConcatDB", "TopK", "TrainParams"]
    if avg_col:
        wide = wide.sort_values(sort_keys + [avg_col], ascending=[True, True, True, True, False], na_position="last")
    else:
        wide = wide.sort_values(sort_keys, ascending=True, na_position="last")

    # Format floats
    for col in ordered:
        wide[col] = wide[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")

    wide = wide.fillna("-").reset_index(drop=True)
    wide.insert(0, "Rank", range(1, len(wide) + 1))

    setting_cols = ["UseCtx", "ConcatDB", "TrainParams", "TopK"]
    # Show short model name + settings columns; Model column contains the compound key
    wide = wide.rename(columns={"ModelName": "BaseModel"})
    return wide[["Rank", "BaseModel", "Count"] + setting_cols + ordered]


def build_long(
    rows: List[Dict],
    dataset=None,
    split=None,
    date=None,
    use_contexts: Optional[str] = None,
    concat_all_db: Optional[bool] = None,
    use_train_params: Optional[bool] = None,
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    """Long format: one row per result file, sorted by dataset/split/EM."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if dataset:          df = df[df["dataset"].str.lower() == dataset.lower()]
    if split:            df = df[df["split"].str.lower() == split.lower()]
    if date:             df = df[df["date"] == date]
    if use_contexts is not None:
        df = df[df["use_contexts"].str.lower() == use_contexts.lower()]
    if concat_all_db is not None:
        df = df[df["concat_all_db"] == concat_all_db]
    if use_train_params is not None:
        df = df[df["use_train_params"] == use_train_params]
    if top_k is not None:
        df = df[df["top_k"] == top_k]
    if df.empty:
        return df

    df = df.sort_values(["dataset", "split", "phase", "em"], ascending=[True, True, True, False])

    df["EM"] = df["em"].apply(lambda x: f"{x:.2f}")
    df["F1"] = df["f1"].apply(lambda x: f"{x:.2f}")
    df["UseCtx"] = df["use_contexts"].fillna("-")
    df["ConcatDB"] = df["concat_all_db"].map(lambda x: "Y" if x is True else ("N" if x is False else "-"))
    df["TrainP"] = df["use_train_params"].map(lambda x: "Y" if x is True else ("N" if x is False else "-"))
    df["TopK"] = df["top_k"].apply(lambda x: str(int(x)) if pd.notna(x) else "-")

    out = df[["dataset", "split", "phase", "date", "model", "count", "UseCtx", "ConcatDB", "TopK", "TrainP", "EM", "F1"]]
    out = out.rename(columns={"dataset": "Dataset", "split": "Split", "phase": "Phase",
                               "date": "Date", "model": "Model", "count": "Count"})
    return out.reset_index(drop=True)


def print_table(df: pd.DataFrame, fmt: str):
    if fmt == "markdown":
        print("\n# Results\n")
        print(df.to_markdown(index=False))
    elif fmt == "long":
        sep = "=" * max(80, 15 * len(df.columns))
        print(f"\n{sep}\nRESULTS (long format)\n{sep}\n")
        print(df.to_string(index=False))
        print(f"\n{sep}")
    else:
        sep = "=" * max(80, 15 * len(df.columns))
        print(f"\n{sep}\nRESULTS SUMMARY\n{sep}\n")
        print(df.to_string(index=False))
        print(f"\n{sep}")


def main():
    parser = argparse.ArgumentParser(description="Summarize multi-hop reasoning results.")
    parser.add_argument("one_phase_root", nargs="?", default=ONE_PHASE_DEFAULT,
                        help="Path to 1phase output root (default: output/lmlm)")
    parser.add_argument("--two_phase_root", default=TWO_PHASE_DEFAULT,
                        help="Path to 2phase output root (default: output/two_phase)")
    parser.add_argument("--dataset", "-d", default=None, help="Filter by dataset")
    parser.add_argument("--split", "-s", default=None, help="Filter by split (dev/train/test)")
    parser.add_argument("--date", default=None, help="Filter by date tag (e.g. 0330)")
    parser.add_argument("--use_contexts", default=None,
                        help="Filter by use_contexts (e.g. golden, all)")
    parser.add_argument("--concat_all_db", default=None, choices=["true", "false"],
                        help="Filter by concat_all_db (true/false)")
    parser.add_argument("--use_train_params", default=None, choices=["true", "false"],
                        help="Filter by use_train_params (true/false)")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Filter by top_k value (e.g. 4)")
    parser.add_argument("--standard", action="store_true",
                        help="Filter to standard setting: use_contexts=all, concat_all_db=true, use_train_params=true, top_k=4")
    parser.add_argument("--format", "-f", default="simple",
                        choices=["simple", "markdown", "long"],
                        help="Output format: simple (wide), markdown (wide), long (one row per result)")
    args = parser.parse_args()

    if args.standard:
        args.use_contexts = args.use_contexts or "all"
        args.concat_all_db = args.concat_all_db or "true"
        args.use_train_params = args.use_train_params or "true"
        args.top_k = args.top_k or 4

    concat_all_db_filter = (
        True if args.concat_all_db == "true" else
        False if args.concat_all_db == "false" else None
    )
    use_train_params_filter = (
        True if args.use_train_params == "true" else
        False if args.use_train_params == "false" else None
    )

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

    filter_kwargs = dict(
        dataset=args.dataset, split=args.split, date=args.date,
        use_contexts=args.use_contexts,
        concat_all_db=concat_all_db_filter,
        use_train_params=use_train_params_filter,
        top_k=args.top_k,
    )

    if args.format == "long":
        df = build_long(rows, **filter_kwargs)
    else:
        df = build_wide(rows, **filter_kwargs)

    if df.empty:
        print("No results match the specified filters.")
        return
    print()
    print_table(df, args.format)


if __name__ == "__main__":
    main()
