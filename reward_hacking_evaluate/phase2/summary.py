"""
Quick summary of Phase 2 results (Grounding + Reasoning only).
Usage: python phase2/summary.py
       python phase2/summary.py --input path/to/results.json
"""

import argparse
import json
import os
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default=os.path.join(os.path.dirname(__file__), "phase2_results.json"))
    args = parser.parse_args()

    with open(args.input, "r") as f:
        results = json.load(f)

    total_rows = len(results)
    grounding_counts = Counter()
    reasoning_counts = Counter()
    total_completions = 0
    evaluated_completions = 0

    for row in results:
        for db in row.get("db_results", []):
            for comp in db.get("completions", []):
                total_completions += 1
                if comp.get("evaluation"):
                    evaluated_completions += 1
                    grounding_counts[comp["grounding"]] += 1
                    reasoning_counts[comp["reasoning"]] += 1

    pct = lambda n, d: f"{n / d:.1%}" if d > 0 else "N/A"

    print(f"{'='*55}")
    print(f"  Phase 2 Results Summary")
    print(f"{'='*55}")
    print(f"  Rows total:              {total_rows}")
    print(f"  Completions total:       {total_completions}")
    print(f"  Completions evaluated:   {evaluated_completions}")

    print(f"\n{'='*55}")
    print(f"  DB Grounding Distribution")
    print(f"{'='*55}")
    for label in ["fully_grounded", "partially_grounded", "ungrounded", "no_answer"]:
        count = grounding_counts.get(label, 0)
        print(f"  {label:<25s} {count:>4d}  ({pct(count, evaluated_completions)})")

    print(f"\n{'='*55}")
    print(f"  Reasoning Quality Distribution")
    print(f"{'='*55}")
    for label in ["correct", "minor_error", "major_error"]:
        count = reasoning_counts.get(label, 0)
        print(f"  {label:<25s} {count:>4d}  ({pct(count, evaluated_completions)})")

    print(f"{'='*55}")


if __name__ == "__main__":
    main()