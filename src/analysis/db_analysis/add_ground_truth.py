"""
Match HotpotQA ground truth answers to test_rollouts.csv.
Requires: pip install datasets

Usage: python analysis/add_ground_truth.py
       python analysis/add_ground_truth.py --csv path/to/csv --output path/to/output.csv
"""

import argparse
import csv
import re
import os

from datasets import load_dataset

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "reward_hacking_evaluate/final_v2.2_375.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "final_v2.2_375_with_gt.csv")


def extract_question(prompt: str) -> str:
    """Strip 'Question: ... Answer: ' wrapper to get the raw question."""
    prompt = prompt.strip()
    # Remove "Question: " prefix
    if prompt.startswith("Question:"):
        prompt = prompt[len("Question:"):].strip()
    # Remove "Answer:" or "Answer: " suffix
    prompt = re.sub(r'\s*Answer:\s*$', '', prompt).strip()
    return prompt


def normalize(text: str) -> str:
    """Normalize text for matching: lowercase, strip, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=CSV_PATH)
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    args = parser.parse_args()

    # ── Load HotpotQA ──
    print("Loading HotpotQA distractor train split...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    print(f"  HotpotQA train size: {len(dataset)}")

    # Build lookup: normalized question -> answer
    qa_lookup = {}
    for item in dataset:
        norm_q = normalize(item["question"])
        qa_lookup[norm_q] = item["answer"]
    print(f"  Unique questions in HotpotQA: {len(qa_lookup)}")

    # ── Load CSV ──
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames)
        rows = list(reader)
    print(f"  CSV rows: {len(rows)}")

    # ── Match ──
    matched = 0
    unmatched = 0
    unmatched_examples = []

    # Add ground_truth_answer column
    if "ground_truth_answer" not in header:
        header.append("ground_truth_answer")

    for row in rows:
        prompt = row.get("phase2_prompt", "")
        raw_question = extract_question(prompt)
        norm_q = normalize(raw_question)

        answer = qa_lookup.get(norm_q, "")
        if answer:
            matched += 1
        else:
            unmatched += 1
            if len(unmatched_examples) < 5:
                unmatched_examples.append(raw_question[:100])

        row["ground_truth_answer"] = answer

    print(f"\n  Matched: {matched}")
    print(f"  Unmatched: {unmatched}")

    if unmatched_examples:
        print(f"\n  Unmatched examples:")
        for ex in unmatched_examples:
            print(f"    {ex}")

    # ── Save ──
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()