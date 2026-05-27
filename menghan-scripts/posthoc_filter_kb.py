"""Post-hoc KB filtering for Table D.

Takes an existing inference CSV and Phase 1 labels, produces a new CSV where
the `generated_db_0` column has had all `hallucinated` triplets removed (per
Phase 1 LLM judge labels). All other CSV columns are preserved verbatim so
the existing evaluate_phase2.py can be re-run on the filtered CSV.

Note: the chain completion (phase2_completion_0_0) is NOT modified — the chain
still references whatever values were originally retrieved. The point of the
re-run is to see if the Phase 2 judge marks those chains as less grounded when
shown the cleaned KB.

Usage:
    /share/j_sun/mx253/envs/lmlm_mx/bin/python menghan-scripts/posthoc_filter_kb.py \\
        --csv KG_results_v2/inference/qwen3_1.7b_sft_n10_sanity.csv \\
        --phase1-json KG_results_v2/phase1_judge/phase1_qwen3_1.7b_sft_n10_sanity.json \\
        --output KG_results_v2/inference/qwen3_1.7b_sft_n10_sanity_filtered.csv
"""

import argparse
import ast
import csv
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV (original inference output)")
    ap.add_argument("--phase1-json", required=True,
                    help="Phase 1 LLM judge JSON, used to identify hallucinated triplets")
    ap.add_argument("--output", required=True, help="Output CSV (filtered)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    p1_path = Path(args.phase1_json)
    out_path = Path(args.output)

    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")
    if not p1_path.exists():
        sys.exit(f"Phase1 JSON not found: {p1_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load phase1 labels: {row_idx: set of hallucinated triplet_indices}
    with open(p1_path) as f:
        p1 = json.load(f)
    hallu_indices_per_row: dict[int, set[int]] = {}
    for r in p1:
        ridx = r["row_index"]
        hallu_set = set()
        for ev in r["db_results"][0]["evaluation"]["triplet_evaluations"]:
            if ev["faithfulness"] == "hallucinated":
                hallu_set.add(ev["triplet_index"])
        hallu_indices_per_row[ridx] = hallu_set

    # Process CSV
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    n_rows_total = len(rows)
    n_rows_filtered = 0
    n_rows_unlabeled = 0
    total_triplets_before = 0
    total_triplets_after = 0
    total_hallu_removed = 0

    with open(out_path, "w", newline="", encoding="utf-8") as outf:
        writer = csv.DictWriter(outf, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for csv_pos, row in enumerate(rows):
            # CSV's `step` is per-batch; row_idx is CSV position. Must match the
            # convention used by run_inference_csv.py when writing JSONL.
            row_idx = csv_pos

            try:
                triplets = ast.literal_eval(row["generated_db_0"])
            except Exception as e:
                print(f"[warn] row {row_idx}: could not parse generated_db_0 ({e}); keeping unchanged")
                writer.writerow(row)
                continue
            total_triplets_before += len(triplets)

            hallu_set = hallu_indices_per_row.get(row_idx)
            if hallu_set is None:
                # Row not labeled (e.g., rows 100-999 if only first 100 labeled).
                # Keep unchanged — we can't filter without labels.
                n_rows_unlabeled += 1
                writer.writerow(row)
                total_triplets_after += len(triplets)
                continue

            filtered = [t for i, t in enumerate(triplets) if i not in hallu_set]
            n_removed = len(triplets) - len(filtered)
            total_triplets_after += len(filtered)
            total_hallu_removed += n_removed
            if n_removed > 0:
                n_rows_filtered += 1

            new_row = dict(row)
            new_row["generated_db_0"] = str(filtered)
            writer.writerow(new_row)

    print(f"== Posthoc KB filter ==")
    print(f"  Input CSV:    {csv_path}")
    print(f"  Phase1 JSON:  {p1_path}")
    print(f"  Output CSV:   {out_path}")
    print(f"  Rows total:           {n_rows_total}")
    print(f"  Rows labeled:         {n_rows_total - n_rows_unlabeled}")
    print(f"  Rows unlabeled (kept as-is): {n_rows_unlabeled}")
    print(f"  Rows with any hallu removed: {n_rows_filtered}")
    print(f"  Triplets before:      {total_triplets_before:,}")
    print(f"  Triplets after:       {total_triplets_after:,}")
    print(f"  Hallucinated removed: {total_hallu_removed:,} "
          f"({total_hallu_removed/total_triplets_before*100:.2f}% of input)")


if __name__ == "__main__":
    main()
