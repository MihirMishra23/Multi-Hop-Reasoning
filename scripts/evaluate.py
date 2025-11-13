#!/usr/bin/env python3
"""CLI to evaluate a preds JSON and write consolidated metrics to results/eval.

Usage:
  python scripts/evaluate.py \
    --preds preds/icl/hotpotqa_distractor_dev_bn=1_bs=3.json \
    --outdir results/eval

Auto-extracts dataset/agent/llm/bn/bs/split from the preds file contents/path.
Allows overriding via flags.
"""

import argparse
import os
import sys
from datetime import datetime


# Ensure imports work when running directly from repo
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.eval.evaluate import (
    evaluate_file,
    build_output_filename,
    save_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate preds JSON and write metrics to results/eval")
    parser.add_argument("--preds", required=True, help="Path to preds JSON file")
    parser.add_argument("--outdir", default=os.path.join(REPO_ROOT, "results", "eval"), help="Output directory")
    # Optional overrides
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--setting", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--source", default="hf")
    parser.add_argument("--agent", default=None)
    parser.add_argument("--llm", default=None)
    parser.add_argument("--bn", type=int, default=None)
    parser.add_argument("--bs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Evaluate
    results = evaluate_file(
        args.preds,
        dataset=args.dataset,
        setting=args.setting,
        split=args.split,
        source=args.source,
    )

    # Build timestamp MM-DD-HH
    timestamp = datetime.now().strftime("%m-%d-%H")

    # Override meta fields if provided
    meta = results["meta"]
    if args.dataset is not None:
        meta["dataset"] = args.dataset
    if args.agent is not None:
        meta["agent"] = args.agent
    if args.llm is not None:
        meta["llm"] = args.llm
    if args.bn is not None:
        meta["bn"] = args.bn
    if args.bs is not None:
        meta["bs"] = args.bs
    if args.dataset is not None:
        meta["dataset"] = args.dataset
    if args.setting is not None:
        meta["setting"] = args.setting
    if args.split is not None:
        meta["split"] = args.split
    if args.seed is not None:
        meta["seed"] = args.seed
    # Save timestamp in the JSON payload
    meta["timestamp"] = timestamp

    filename = build_output_filename(
        dataset=meta.get("dataset", "unknown"),
        agent=meta.get("agent", "unknown"),
        llm=meta.get("llm", "unknown"),
        bn=int(meta.get("bn", -1)),
        bs=int(meta.get("bs", -1)),
        timestamp=timestamp,
    )

    outpath = save_results(results, args.outdir, filename)
    print(outpath)


if __name__ == "__main__":
    main()


