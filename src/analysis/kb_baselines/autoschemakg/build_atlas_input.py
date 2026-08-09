"""Convert a multi-hop QA dataset (HotpotQA/MuSiQue/2Wiki) into the input JSON
format that atlas-rag's KnowledgeGraphExtractor expects.

atlas-rag input format (list of objects):
    [{"id": "<unique_id>", "text": "<passage text>", "metadata": {...}}, ...]

We emit one record per passage (not per question), so retrieval downstream is
passage-level — same granularity as KBevo Phase-1.

ID convention: ``{qid}__{para_idx}`` keeps qid + para index recoverable.

Usage:
    PYTHONPATH=src python kb_baselines/autoschemakg/build_atlas_input.py \
        --dataset hotpotqa --split dev --setting distractor \
        --n 5 --seed 42 \
        --output kb_baselines/autoschemakg/output/inputs/hotpotqa_dev_n5_seed42.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import get_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["hotpotqa", "musique", "2wiki"])
    p.add_argument("--split", default="dev")
    p.add_argument("--setting", default="distractor")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-contexts", default="all", choices=["all", "golden"])
    p.add_argument("--output", required=True)
    args = p.parse_args()

    ds = get_dataset(name=args.dataset, setting=args.setting, split=args.split, seed=args.seed)
    end = min(args.start_index + args.n, len(ds))
    ds = ds.select(range(args.start_index, end))
    print(f"Loaded {len(ds)} questions from {args.dataset}/{args.split} (seed={args.seed})")

    records = []
    for ex in ds:
        qid = ex.get("id") or ex.get("_id") or ex.get("case_id")
        contexts = ex["golden_contexts"] if args.use_contexts == "golden" else ex["contexts"]
        for i, ctx in enumerate(contexts):
            records.append({
                "id": f"{qid}__{i}",
                "text": ctx,
                "metadata": {"qid": str(qid), "para_idx": i, "lang": "en"},
            })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(records)} passages -> {args.output}")
    print(f"   ({len(records)/len(ds):.1f} passages/question avg)")


if __name__ == "__main__":
    main()
