"""
Phase 2: Lookup Efficiency Analysis (code-only, no API calls)
Usage:
    python phase2/lookup_efficiency.py --num_rows 10
    python phase2/lookup_efficiency.py
"""

import argparse
import ast
import csv
import json
import random
import re
import os
from pathlib import Path


# ── Config ──────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "final_v2.2_0.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "lookup_efficiency_results.json")


# ── Helpers ─────────────────────────────────────────────────────────────
def get_db_columns(header: list[str]) -> list[str]:
    db_cols = [c for c in header if c.startswith("generated_db_")]
    db_cols.sort(key=lambda c: int(c.split("_")[-1]))
    return db_cols


def get_completion_columns(header: list[str], db_idx: int) -> list[str]:
    prefix = f"phase2_completion_{db_idx}_"
    comp_cols = [c for c in header if c.startswith(prefix)]
    comp_cols.sort(key=lambda c: int(c.split("_")[-1]))
    return comp_cols


def parse_lookups(completion: str) -> list[dict]:
    """Parse all DB lookups from a completion."""
    pattern = r'<\|db_entity\|>\s*(.*?)\s*<\|db_relationship\|>\s*(.*?)\s*<\|db_return\|>\s*(.*?)\s*<\|db_end\|>'
    matches = re.findall(pattern, completion)
    lookups = []
    for entity, relationship, return_val in matches:
        lookups.append({
            "entity": entity.strip(),
            "relationship": relationship.strip(),
            "return_value": return_val.strip(),
        })
    return lookups


def compute_efficiency(lookups: list[dict]) -> dict:
    """Compute lookup efficiency metrics."""
    total = len(lookups)
    if total == 0:
        return {
            "total_lookups": 0,
            "unique_lookups": 0,
            "redundancy_rate": 0.0,
            "unknown_rate": 0.0,
            "max_consecutive_unknown": 0,
            "efficiency_score": 1.0,
        }

    unique_pairs = set()
    for l in lookups:
        unique_pairs.add((l["entity"], l["relationship"]))
    unique = len(unique_pairs)

    unknown_count = sum(1 for l in lookups if l["return_value"].lower() == "unknown")

    max_consec = 0
    current_consec = 0
    for l in lookups:
        if l["return_value"].lower() == "unknown":
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0

    return {
        "total_lookups": total,
        "unique_lookups": unique,
        "redundancy_rate": 1 - (unique / total) if total > 0 else 0.0,
        "unknown_rate": unknown_count / total if total > 0 else 0.0,
        "max_consecutive_unknown": max_consec,
        "efficiency_score": unique / total if total > 0 else 1.0,
    }


# ── Main ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 2: Lookup Efficiency Analysis")
    parser.add_argument("--num_rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv", type=str, default=CSV_PATH)
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    args = parser.parse_args()

    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        db_cols = get_db_columns(header)
        rows = list(reader)

    if args.num_rows is not None and args.num_rows < len(rows):
        random.seed(args.seed)
        rows = random.sample(rows, args.num_rows)
        print(f"Randomly sampled {args.num_rows} rows (seed={args.seed})")
    else:
        print(f"Evaluating all {len(rows)} rows")

    all_results = []
    # Aggregate stats
    all_efficiency = []
    all_redundancy = []
    all_total = []
    all_unknown = []
    all_consec = []

    for row_idx, row in enumerate(rows):
        row_result = {
            "row_index": row_idx,
            "completions": [],
        }

        for db_col in db_cols:
            raw_db = row.get(db_col, "")
            if not raw_db.strip():
                continue

            db_idx = int(db_col.split("_")[-1])
            comp_cols = get_completion_columns(header, db_idx)

            for comp_col in comp_cols:
                raw_comp = row.get(comp_col, "")
                if not raw_comp.strip():
                    continue

                lookups = parse_lookups(raw_comp)
                efficiency = compute_efficiency(lookups)

                row_result["completions"].append({
                    "db_column": db_col,
                    "completion_column": comp_col,
                    "efficiency": efficiency,
                })

                if efficiency["total_lookups"] > 0:
                    all_efficiency.append(efficiency["efficiency_score"])
                    all_redundancy.append(efficiency["redundancy_rate"])
                    all_total.append(efficiency["total_lookups"])
                    all_unknown.append(efficiency["unknown_rate"])
                    all_consec.append(efficiency["max_consecutive_unknown"])

        all_results.append(row_result)

    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {args.output}")

    # Print summary
    avg = lambda lst: sum(lst) / len(lst) if lst else 0
    pct = lambda n, d: f"{n / d:.1%}" if d > 0 else "N/A"
    total_comps = sum(len(r["completions"]) for r in all_results)

    print(f"\n{'='*55}")
    print(f"  Lookup Efficiency Summary")
    print(f"{'='*55}")
    print(f"  Completions analyzed:    {total_comps}")
    print(f"  With lookups:            {len(all_efficiency)}")
    if all_efficiency:
        print(f"\n  Avg efficiency score:    {avg(all_efficiency):.2%}")
        print(f"  Avg redundancy rate:     {avg(all_redundancy):.2%}")
        print(f"  Avg total lookups:       {avg(all_total):.1f}")
        print(f"  Avg unknown rate:        {avg(all_unknown):.2%}")
        print(f"  Avg max consec unknown:  {avg(all_consec):.1f}")
        high_redundancy = sum(1 for r in all_redundancy if r > 0.5)
        print(f"\n  Redundancy >50%:         {high_redundancy} / {len(all_redundancy)} ({pct(high_redundancy, len(all_redundancy))})")
        no_lookup = total_comps - len(all_efficiency)
        print(f"  No lookups at all:       {no_lookup} / {total_comps} ({pct(no_lookup, total_comps)})")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()