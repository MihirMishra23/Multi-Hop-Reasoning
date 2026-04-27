#!/usr/bin/env python3
"""Check whether contexts in a CSV are golden (2 paragraphs) or distractor (10 paragraphs)
by comparing against the HotpotQA dataset.

Usage:
    PYTHONPATH=src python menghan-scripts/check_context_type.py \
        --csv KG_results/Qwen3-4B-SFT_...csv \
        --n 5
"""

import argparse
import csv
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from data import get_dataset


def extract_question(phase2_prompt: str) -> str:
    """Extract question from 'Question:\n{q}\nAnswer:\n'"""
    try:
        return phase2_prompt.split("Question:\n")[1].split("\nAnswer:\n")[0].strip()
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--n", type=int, default=5, help="Number of examples to check")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--setting", default="distractor")
    args = parser.parse_args()

    # Load dataset to get golden + distractor contexts
    print("Loading HotpotQA dataset...")
    ds = get_dataset(name="hotpotqa", setting=args.setting, split=args.split)

    # Build lookup: question -> example
    q_to_ex = {}
    for ex in ds:
        q_to_ex[ex["question"].strip()] = ex

    # Read CSV
    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if len(rows) >= args.n:
                break

    print(f"\nChecking {len(rows)} examples from: {args.csv}\n")
    print("=" * 80)

    for i, row in enumerate(rows):
        question = extract_question(row["phase2_prompt"])
        csv_context = row["phase1_context_0"]

        ex = q_to_ex.get(question)
        if ex is None:
            print(f"[{i}] Question not found in dataset: {question[:80]}")
            continue

        golden_contexts = ex.get("golden_contexts", [])
        all_contexts = ex.get("contexts", [])

        golden_joined = "\n\n".join(golden_contexts)
        all_joined = "\n\n".join(all_contexts)

        is_golden = (csv_context.strip() == golden_joined.strip())
        is_all = (csv_context.strip() == all_joined.strip())

        # Count how many paragraphs are in the csv context
        # (rough count by splitting on double newline)
        csv_paras = [p for p in csv_context.split("\n\n") if p.strip()]

        print(f"[{i}] Q: {question[:70]}")
        print(f"     CSV context paragraphs : {len(csv_paras)}")
        print(f"     Golden contexts count  : {len(golden_contexts)}")
        print(f"     All contexts count     : {len(all_contexts)}")
        print(f"     Matches golden?        : {is_golden}")
        print(f"     Matches all?           : {is_all}")
        if not is_golden and not is_all:
            # Check partial overlap
            csv_paras_set = set(p.strip() for p in csv_paras)
            golden_set = set(p.strip() for p in golden_contexts)
            all_set = set(p.strip() for p in all_contexts)
            golden_overlap = len(csv_paras_set & golden_set)
            all_overlap = len(csv_paras_set & all_set)
            print(f"     Overlap with golden    : {golden_overlap}/{len(golden_set)}")
            print(f"     Overlap with all       : {all_overlap}/{len(all_set)}")
        print()


if __name__ == "__main__":
    main()
