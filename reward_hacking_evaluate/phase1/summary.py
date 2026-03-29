"""
Quick summary of Phase 1 results.
Usage: python phase1/summary.py
       python phase1/summary.py --input path/to/results.json
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default=os.path.join(os.path.dirname(__file__), "phase1_results.json"))
    args = parser.parse_args()

    with open(args.input, "r") as f:
        results = json.load(f)

    total_rows = len(results)
    valid_rows = [r for r in results if r["row_score"] > 0]

    # Collect all DB-level scores and triplet-level details
    all_faithfulness = []
    all_quality = []
    all_overall = []
    total_dbs = 0
    total_triplets = 0
    total_hallucinated = 0
    total_clean = 0
    total_with_quality_issues = 0
    quality_issue_counts = Counter()

    for row in results:
        for db in row.get("db_results", []):
            s = db["scores"]
            all_faithfulness.append(s["faithfulness_score"])
            all_quality.append(s["quality_score"])
            all_overall.append(s["overall_score"])
            total_dbs += 1
            total_triplets += s["total_triplets"]
            total_hallucinated += s["hallucinated_count"]
            total_clean += s["clean_count"]

            # Count per-triplet quality issues
            for triplet_eval in db["evaluation"]["triplet_evaluations"]:
                issues = triplet_eval.get("quality_issues", [])
                if triplet_eval.get("faithfulness") == "faithful" and len(issues) > 0:
                    total_with_quality_issues += 1
                for issue in issues:
                    quality_issue_counts[issue] += 1

    avg = lambda lst: sum(lst) / len(lst) if lst else 0
    pct = lambda n, d: f"{n / d:.1%}" if d > 0 else "N/A"

    print(f"{'='*55}")
    print(f"  Phase 1 Results Summary")
    print(f"{'='*55}")
    print(f"  Rows total:          {total_rows}")
    print(f"  Rows evaluated:      {len(valid_rows)}")
    print(f"  DBs evaluated:       {total_dbs}")
    print(f"  Triplets total:      {total_triplets}")

    print(f"\n{'='*55}")
    print(f"  Triplet-level Breakdown")
    print(f"{'='*55}")
    print(f"  Clean (score=1):     {total_clean} / {total_triplets} ({pct(total_clean, total_triplets)})")
    print(f"  Hallucinated:        {total_hallucinated} / {total_triplets} ({pct(total_hallucinated, total_triplets)})")
    print(f"  Quality issues:      {total_with_quality_issues} / {total_triplets} ({pct(total_with_quality_issues, total_triplets)})")

    print(f"\n{'='*55}")
    print(f"  Quality Issue Breakdown (among all triplets)")
    print(f"{'='*55}")
    all_issue_types = [
        "ambiguous_entity_value",
        "trivial",
        "non_specific",
        "malformed",
        "reversed_roles",
    ]
    for issue_type in all_issue_types:
        count = quality_issue_counts.get(issue_type, 0)
        print(f"  {issue_type:<30s} {count:>4d}  ({pct(count, total_triplets)})")

    # Also show any unexpected issue types
    unexpected = set(quality_issue_counts.keys()) - set(all_issue_types)
    for issue_type in sorted(unexpected):
        count = quality_issue_counts[issue_type]
        print(f"  {issue_type:<30s} {count:>4d}  ({pct(count, total_triplets)})  [unexpected]")

    print(f"\n{'='*55}")
    print(f"  Aggregate Scores (DB-level avg)")
    print(f"{'='*55}")
    print(f"  Avg Faithfulness:    {avg(all_faithfulness):.2%}")
    print(f"  Avg Quality:         {avg(all_quality):.2%}")
    print(f"  Avg Overall:         {avg(all_overall):.2%}")

    print(f"\n{'='*55}")
    print(f"  Final Score (Row-level avg)")
    print(f"{'='*55}")
    print(f"  Row-level score:     {avg([r['row_score'] for r in valid_rows]):.2%}")
    print(f"  (1.0 = perfect, 0.0 = worst)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()