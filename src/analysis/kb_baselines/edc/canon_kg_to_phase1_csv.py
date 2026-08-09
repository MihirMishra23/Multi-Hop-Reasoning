"""Convert EDC canon_kg.txt (per-passage) → per-passage CSV that
phase1 judge (`reward_hacking_evaluate/phase1/evaluate_phase1.py`) expects.

Output CSV columns:
    phase1_context_0  : the passage text
    generated_db_0    : Python literal list of [s, r, o] triplets for that passage
                        (entities normalized to natural-language: _→space)

One row per passage (NOT per question). The phase1 judge then samples N rows
(e.g. 100) with seed for reproducibility.

Usage:
    PYTHONPATH=src python kb_baselines/edc/canon_kg_to_phase1_csv.py \\
        --canon-kg /share/j_sun/mx253/edc/output/hotpotqa_dev_n1000/iter0/canon_kg.txt \\
        --passages-txt /share/j_sun/mx253/edc/datasets/hotpotqa.txt \\
        --output db_analysis/edc_off_mistral7b_hotpotqa_per_passage.csv
"""

import argparse
import ast
import csv
import os


def _entity_norm(s: str) -> str:
    return s.replace("_", " ").strip()


def parse_canon_kg(path):
    per_passage = []
    with open(path) as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw.strip():
                per_passage.append([])
                continue
            try:
                lst = ast.literal_eval(raw)
            except Exception:
                per_passage.append([])
                continue
            kept = []
            for t in lst or []:
                if (isinstance(t, (list, tuple)) and len(t) >= 3
                        and all(isinstance(x, str) for x in t[:3])):
                    kept.append([_entity_norm(t[0]), t[1].strip(), _entity_norm(t[2])])
            per_passage.append(kept)
    return per_passage


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--canon-kg", required=True)
    p.add_argument("--passages-txt", required=True,
                   help="line-aligned with canon_kg.txt — original passage texts")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print(f"loading canon_kg from {args.canon_kg}")
    per_passage_triplets = parse_canon_kg(args.canon_kg)
    print(f"  {len(per_passage_triplets)} passages, "
          f"{sum(len(x) for x in per_passage_triplets)} total triplets")

    print(f"loading passages from {args.passages_txt}")
    with open(args.passages_txt) as f:
        passages = [line.rstrip("\n") for line in f]
    print(f"  {len(passages)} passages")
    assert len(passages) == len(per_passage_triplets), (
        f"misalign: passages={len(passages)} vs triplets={len(per_passage_triplets)}"
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase1_context_0", "generated_db_0"])
        n_written = n_skipped = 0
        for ctx, triplets in zip(passages, per_passage_triplets):
            if not ctx.strip() or not triplets:
                n_skipped += 1
                continue
            w.writerow([ctx, repr(triplets)])
            n_written += 1
    print(f"wrote {n_written} rows to {args.output}  (skipped {n_skipped} empty)")


if __name__ == "__main__":
    main()
