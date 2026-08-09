"""Convert EDC canon_kg.txt → per-question CSV that db_analysis/coverage_analysis.py
expects (for hop-reachability metric in Table R4-style).

Output CSV columns (matching coverage_analysis.py's `row.get(...)` calls):
    phase1_context_0      : qid (used only for dedup; guaranteed unique per row)
    generated_db_0        : Python literal list of [s, r, o] triplets — UNION
                            of all triplets from passages belonging to this qid
    phase2_prompt         : "Question: {q}\\nAnswer:"  (script extract_question parses this)
    ground_truth_answer   : gold answer string

Usage:
    PYTHONPATH=src python kb_baselines/edc/canon_kg_to_coverage_csv.py \\
        --canon-kg /share/j_sun/mx253/edc/output/hotpotqa_dev_n1000/iter0/canon_kg.txt \\
        --ids-json /share/j_sun/mx253/edc/datasets/hotpotqa.ids.json \\
        --dataset hotpotqa --n 1000 --seed 42 \\
        --output db_analysis/edc_official_mistral7b_hotpotqa_dev_n1000_seed42_coverage_input.csv
"""

import argparse
import ast
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from data import get_dataset


def _entity_norm(s: str) -> str:
    """EDC OIE (with webnlg-style few-shot) emits entities with underscores
    (e.g. 'Robin_Barton', 'Hollywood_Pictures'). coverage_analysis.normalize()
    uses `\\w` which preserves underscores → substring match fails against
    natural-language questions/answers ('robin barton'). Replace underscores
    with spaces in subject + object so reachability is comparable to ATLAS/KBevo
    KBs (which use natural-language entities).
    """
    return s.replace("_", " ").strip()


def parse_canon_kg(path):
    """Read canon_kg.txt → list of per-passage triplet lists (same order as input .txt)."""
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
                    # underscore→space on subject + object (relation stays as label)
                    kept.append([_entity_norm(t[0]), t[1].strip(), _entity_norm(t[2])])
            per_passage.append(kept)
    return per_passage


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--canon-kg", required=True)
    p.add_argument("--ids-json", required=True,
                   help="passage→qid mapping (matches canon_kg row order)")
    p.add_argument("--dataset", required=True, choices=["hotpotqa", "musique", "2wiki"])
    p.add_argument("--split", default="dev")
    p.add_argument("--setting", default="distractor")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print(f"Loading canon_kg from {args.canon_kg} ...")
    per_passage = parse_canon_kg(args.canon_kg)
    print(f"  {len(per_passage)} passages, "
          f"{sum(len(x) for x in per_passage)} raw triplets")

    print(f"Loading ids mapping from {args.ids_json} ...")
    with open(args.ids_json) as f:
        ids = json.load(f)
    assert len(ids) == len(per_passage), (
        f"length mismatch: ids={len(ids)} vs canon_kg passages={len(per_passage)}"
    )

    # Group triplets by qid, deduping within
    qid_to_triplets = defaultdict(list)
    qid_to_seen = defaultdict(set)
    for passage_meta, triplets in zip(ids, per_passage):
        qid = passage_meta["qid"]
        for t in triplets:
            key = (t[0], t[1], t[2])
            if key in qid_to_seen[qid]:
                continue
            qid_to_seen[qid].add(key)
            qid_to_triplets[qid].append(t)

    print(f"  unique qids: {len(qid_to_triplets)}")
    print(f"  triplets/qid: min={min(len(v) for v in qid_to_triplets.values())} "
          f"max={max(len(v) for v in qid_to_triplets.values())} "
          f"mean={sum(len(v) for v in qid_to_triplets.values())/len(qid_to_triplets):.0f}")

    print(f"Loading dataset {args.dataset}/{args.split} (n={args.n}, seed={args.seed}) "
          f"to get questions + answers ...")
    ds = get_dataset(name=args.dataset, setting=args.setting,
                     split=args.split, seed=args.seed)
    ds = ds.select(range(args.n))
    print(f"  {len(ds)} questions loaded")

    # Write CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    rows_written = 0
    missing_kb = 0
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase1_context_0", "generated_db_0",
                    "phase2_prompt", "ground_truth_answer"])
        for ex in ds:
            qid = ex.get("id") or ex.get("_id") or ex.get("case_id")
            qid = str(qid)
            question = ex.get("question", "")
            answers = ex.get("answers", [])
            gt = (answers[0] if isinstance(answers, list) and answers else
                  str(answers) if answers else "")
            triplets = qid_to_triplets.get(qid, [])
            if not triplets:
                missing_kb += 1
                continue
            w.writerow([
                qid,                                            # phase1_context_0 (dedup key)
                repr(triplets),                                 # generated_db_0 (literal list)
                f"Question: {question}\nAnswer:",               # phase2_prompt
                gt,                                              # ground_truth_answer
            ])
            rows_written += 1
    print(f"Wrote {rows_written} rows to {args.output}  (missing KB for {missing_kb})")


if __name__ == "__main__":
    main()
