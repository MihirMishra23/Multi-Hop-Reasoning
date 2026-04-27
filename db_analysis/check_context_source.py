"""
Check if contexts in CSV contain only gold (supporting) paragraphs
or also distractor paragraphs from HotpotQA.

Usage: python analysis/check_context_source.py
"""

import csv
import re
import os
from datasets import load_dataset

CSV_PATH = "/share/j_sun/mx253/Multi-Hop-Reasoning/KG_results/ckpt500_hotpotqa_dev_n1000_all_concat_trainparams.csv"


def normalize(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def extract_question(prompt):
    prompt = prompt.strip()
    if prompt.startswith("Question:"):
        prompt = prompt[len("Question:"):].strip()
    prompt = re.sub(r'\s*Answer:\s*$', '', prompt).strip()
    return prompt


def main():
    # Load CSV
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    print(f"CSV rows: {len(rows)}")
    print(f"Columns: {[c for c in header if 'context' in c.lower() or 'prompt' in c.lower()][:10]}")

    # Load HotpotQA - both splits
    print("\nLoading HotpotQA...")
    qa_lookup = {}
    for split_name in ["train", "validation"]:
        dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split_name)
        count = 0
        for item in dataset:
            norm_q = normalize(item["question"])
            if norm_q not in qa_lookup:
                qa_lookup[norm_q] = item
                count += 1
        print(f"  {split_name}: {count} questions loaded")
    print(f"  Total: {len(qa_lookup)} unique questions")

    # Check each row
    gold_only = 0
    has_distractors = 0
    has_all = 0
    unmatched = 0
    total = 0

    for row_idx, row in enumerate(rows[:20]):  # Check first 20 for speed
        prompt = row.get("phase2_prompt", "") or row.get("prompt", "")
        context = row.get("phase1_context_0", "") or row.get("context", "")

        if not prompt or not context:
            # Try to find the right column
            if row_idx == 0:
                print(f"\nAvailable columns: {list(row.keys())[:20]}")
            continue

        question = extract_question(prompt)
        norm_q = normalize(question)
        total += 1

        hotpot_item = qa_lookup.get(norm_q)
        if not hotpot_item:
            unmatched += 1
            if unmatched <= 3:
                print(f"  [UNMATCHED] '{question[:80]}'")
                print(f"    normalized: '{norm_q[:80]}'")
            continue

        # HotpotQA has 'context' with titles and sentences
        # supporting_facts tells us which are gold
        supporting_titles = set(hotpot_item["supporting_facts"]["title"])
        all_titles = hotpot_item["context"]["title"]

        context_norm = normalize(context)

        # Check which titles appear in the CSV context
        gold_found = 0
        distractor_found = 0
        for title in all_titles:
            title_norm = normalize(title)
            if title_norm in context_norm:
                if title in supporting_titles:
                    gold_found += 1
                else:
                    distractor_found += 1

        if row_idx < 5:
            print(f"\n{'='*50}")
            print(f"Row {row_idx}")
            print(f"Question: {question[:80]}")
            print(f"Supporting titles: {supporting_titles}")
            print(f"All titles ({len(all_titles)}): {all_titles}")
            print(f"Gold found in context: {gold_found}/{len(supporting_titles)}")
            print(f"Distractors found: {distractor_found}/{len(all_titles) - len(supporting_titles)}")
            print(f"Context preview: {context[:200]}")

        if distractor_found == 0 and gold_found > 0:
            gold_only += 1
        elif distractor_found > 0 and gold_found > 0:
            if distractor_found + gold_found == len(all_titles):
                has_all += 1
            else:
                has_distractors += 1

    print(f"\n{'='*50}")
    print(f"  Context Source Analysis (first {total} rows)")
    print(f"{'='*50}")
    print(f"  Total checked:       {total}")
    print(f"  Unmatched:           {unmatched}")
    print(f"  Gold only:           {gold_only}")
    print(f"  Gold + some distractor: {has_distractors}")
    print(f"  All paragraphs:      {has_all}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()