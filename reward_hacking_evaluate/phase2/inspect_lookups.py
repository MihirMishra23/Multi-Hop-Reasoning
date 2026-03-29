"""
Inspect lookup examples — show worst redundancy cases and no-lookup cases.
Usage:
    python phase2/inspect_lookups.py
    python phase2/inspect_lookups.py --top 5
"""

import argparse
import csv
import re
import os
from collections import Counter


CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "test_rollouts_22316.csv")


def get_db_columns(header):
    db_cols = [c for c in header if c.startswith("generated_db_")]
    db_cols.sort(key=lambda c: int(c.split("_")[-1]))
    return db_cols


def get_completion_columns(header, db_idx):
    prefix = f"phase2_completion_{db_idx}_"
    comp_cols = [c for c in header if c.startswith(prefix)]
    comp_cols.sort(key=lambda c: int(c.split("_")[-1]))
    return comp_cols


def parse_lookups(completion):
    pattern = r'<\|db_entity\|>\s*(.*?)\s*<\|db_relationship\|>\s*(.*?)\s*<\|db_return\|>\s*(.*?)\s*<\|db_end\|>'
    matches = re.findall(pattern, completion)
    return [(e.strip(), r.strip(), v.strip()) for e, r, v in matches]


def clean_completion(text):
    text = re.sub(r'<\|endoftext\|>', '', text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=CSV_PATH)
    parser.add_argument("--top", type=int, default=3, help="Number of examples per category")
    args = parser.parse_args()

    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        db_cols = get_db_columns(header)
        rows = list(reader)

    all_cases = []
    for row_idx, row in enumerate(rows):
        question = row.get("phase2_prompt", "")
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
                total = len(lookups)
                unique_pairs = set((e, r) for e, r, v in lookups)
                unique = len(unique_pairs)
                redundancy = 1 - (unique / total) if total > 0 else 0.0

                # Count duplicates
                pair_counts = Counter((e, r) for e, r, v in lookups)
                duplicates = {k: v for k, v in pair_counts.items() if v > 1}

                all_cases.append({
                    "row": row_idx,
                    "comp_col": comp_col,
                    "question": question,
                    "total": total,
                    "unique": unique,
                    "redundancy": redundancy,
                    "duplicates": duplicates,
                    "lookups": lookups,
                    "full_completion": clean_completion(raw_comp),
                })

    # ── Output dir ──
    output_dir = os.path.join(os.path.dirname(__file__), "examples")
    os.makedirs(output_dir, exist_ok=True)

    # ── No lookups ──
    no_lookup_cases = [c for c in all_cases if c["total"] == 0]
    print(f"{'='*70}")
    print(f"  NO LOOKUP CASES ({len(no_lookup_cases)} total)")
    print(f"{'='*70}")
    for i, case in enumerate(no_lookup_cases[:args.top]):
        filename = f"no_lookup_{i}.txt"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Row: {case['row']}\n")
            f.write(f"Column: {case['comp_col']}\n")
            f.write(f"Question: {case['question']}\n")
            f.write(f"Total lookups: 0\n")
            f.write(f"\n{'='*60}\n")
            f.write(f"FULL COMPLETION:\n{'='*60}\n\n")
            f.write(case["full_completion"])
        print(f"  [{i}] Row {case['row']} | {case['comp_col']} → saved to {filepath}")

    # ── Worst redundancy ──
    redundant_cases = sorted(
        [c for c in all_cases if c["total"] > 0],
        key=lambda c: c["redundancy"],
        reverse=True,
    )
    print(f"\n{'='*70}")
    print(f"  WORST REDUNDANCY CASES (top {args.top})")
    print(f"{'='*70}")
    for i, case in enumerate(redundant_cases[:args.top]):
        filename = f"redundant_{i}.txt"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Row: {case['row']}\n")
            f.write(f"Column: {case['comp_col']}\n")
            f.write(f"Question: {case['question']}\n")
            f.write(f"Total lookups: {case['total']} | Unique: {case['unique']} | Redundancy: {case['redundancy']:.1%}\n")
            f.write(f"\nRepeated queries:\n")
            for (entity, rel), count in sorted(case["duplicates"].items(), key=lambda x: -x[1]):
                f.write(f"  ({entity}, {rel}) — repeated {count}x\n")
            f.write(f"\n{'='*60}\n")
            f.write(f"FULL COMPLETION:\n{'='*60}\n\n")
            f.write(case["full_completion"])
        print(f"  [{i}] Row {case['row']} | {case['comp_col']} | redundancy={case['redundancy']:.1%} | {case['total']} lookups → saved to {filepath}")

    # ── Most lookups ──
    most_lookups = sorted(all_cases, key=lambda c: c["total"], reverse=True)
    print(f"\n{'='*70}")
    print(f"  MOST LOOKUPS (top {args.top})")
    print(f"{'='*70}")
    for i, case in enumerate(most_lookups[:args.top]):
        filename = f"most_lookups_{i}.txt"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"Row: {case['row']}\n")
            f.write(f"Column: {case['comp_col']}\n")
            f.write(f"Question: {case['question']}\n")
            f.write(f"Total lookups: {case['total']} | Unique: {case['unique']} | Redundancy: {case['redundancy']:.1%}\n")
            if case["duplicates"]:
                f.write(f"\nRepeated queries:\n")
                for (entity, rel), count in sorted(case["duplicates"].items(), key=lambda x: -x[1]):
                    f.write(f"  ({entity}, {rel}) — repeated {count}x\n")
            f.write(f"\n{'='*60}\n")
            f.write(f"FULL COMPLETION:\n{'='*60}\n\n")
            f.write(case["full_completion"])
        print(f"  [{i}] Row {case['row']} | {case['comp_col']} | {case['total']} lookups → saved to {filepath}")

    print(f"\nAll examples saved to {output_dir}/")


if __name__ == "__main__":
    main()