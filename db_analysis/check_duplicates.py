"""
Check for duplicate contexts in test_rollouts.csv
Usage: python analysis/check_duplicates.py
"""

import csv
import os
from collections import Counter

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "reward_hacking_evaluate/final_v2.2_375.csv")


def main():
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Total rows: {len(rows)}")

    # Check context duplicates
    contexts = []
    for row in rows:
        ctx = row.get("phase1_context_0", "").strip()
        if ctx:
            contexts.append(ctx)

    print(f"Rows with context: {len(contexts)}")
    print(f"Unique contexts: {len(set(contexts))}")
    print(f"Duplicate contexts: {len(contexts) - len(set(contexts))}")

    if len(contexts) != len(set(contexts)):
        # Show which contexts are duplicated
        ctx_counts = Counter(contexts)
        duplicates = {k: v for k, v in ctx_counts.items() if v > 1}
        print(f"\n{'='*55}")
        print(f"  Duplicated Contexts ({len(duplicates)} unique)")
        print(f"{'='*55}")
        for ctx, count in sorted(duplicates.items(), key=lambda x: -x[1])[:20]:
            print(f"  Count: {count} | Context preview: {ctx[:100]}...")
    else:
        print("\nNo duplicate contexts found.")

    # Also check phase2_prompt duplicates
    prompts = []
    for row in rows:
        p = row.get("phase2_prompt", "").strip()
        if p:
            prompts.append(p)

    print(f"\n{'='*55}")
    print(f"  Phase 2 Prompts")
    print(f"{'='*55}")
    print(f"Rows with prompt: {len(prompts)}")
    print(f"Unique prompts: {len(set(prompts))}")
    print(f"Duplicate prompts: {len(prompts) - len(set(prompts))}")


if __name__ == "__main__":
    main()