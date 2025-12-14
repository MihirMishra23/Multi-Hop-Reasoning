#!/usr/bin/env python3
"""CLI to evaluate preds JSON files and write consolidated metrics to results/eval.

Usage:
  # Evaluate a single file
  python scripts/evaluate.py \
    --preds preds/icl/hotpotqa_distractor/gpt-4/validation_seed=0_bn=1_bs=3.json \
    --outdir results/eval

  # Evaluate multiple batch files using a pattern
  python scripts/evaluate.py \
    --pattern "preds/icl/hotpotqa_distractor/gpt-4/validation_seed=0_bn=*_bs=2.json" \
    --outdir results/eval

  # Evaluate using directory-based approach
  python scripts/evaluate.py \
    --input-dir preds/icl/hotpotqa_distractor/gpt-4 \
    --split validation \
    --seed 0 \
    --batch-size 2 \
    --outdir results/eval

Auto-extracts dataset/agent/llm/bn/bs/split from the preds file contents/path.
Allows overriding via flags.
The output evaluation file uses the same naming convention as aggregate_batches.py
with question ranges: {split}_seed={seed}_bs={batch_size}_q{first_q}-{last_q}_evaluation.json
"""

import argparse
import json
import glob
import os
import sys
import re
import tempfile
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple


# Ensure imports work when running directly from repo
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from eval.evaluate import (
    evaluate_file,
    build_output_filename,
    save_results,
)


def extract_batch_info(filename: str) -> Tuple[Optional[int], Optional[int]]:
    """Extract batch number and batch size from filename.

    Returns (batch_number, batch_size) or (None, None) if not found.
    """
    pattern = r"_bn=(\d+)_bs=(\d+)\.json$"
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def calculate_question_range(batch_numbers: List[int], batch_size: int) -> Tuple[int, int]:
    """Calculate first and last question numbers from batch numbers and batch size.

    Assumes 1-indexed question numbers.
    Batch n with batch_size bs contains questions: (n-1)*bs + 1 to n*bs
    """
    if not batch_numbers:
        raise ValueError("Cannot calculate question range from empty batch list")

    first_batch = min(batch_numbers)
    last_batch = max(batch_numbers)

    # First question: (first_batch - 1) * batch_size + 1
    first_question = (first_batch - 1) * batch_size + 1
    # Last question: last_batch * batch_size
    last_question = last_batch * batch_size

    return first_question, last_question


def extract_metadata_from_pred_file(pred_file: str) -> Dict[str, Any]:
    """Extract metadata from a pred file."""
    with open(pred_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("metadata", {})


def aggregate_batch_files(batch_files: List[str]) -> Dict[str, Any]:
    """Aggregate multiple batch files into a single data structure."""
    if not batch_files:
        raise ValueError("No batch files provided")

    # Load first file to get metadata structure
    with open(batch_files[0], "r", encoding="utf-8") as f:
        first_batch = json.load(f)

    # Aggregate results
    all_results: Dict[str, Dict[str, Any]] = {}
    total_examples = 0

    for batch_file in batch_files:
        with open(batch_file, "r", encoding="utf-8") as f:
            batch_data = json.load(f)

        batch_results = batch_data.get("results", {})
        all_results.update(batch_results)
        total_examples += len(batch_results)

    # Create aggregated output
    aggregated = {
        "metadata": {
            **first_batch["metadata"],
            "batch_size": total_examples,
            "batch_number": 1,  # Aggregated is treated as single batch
        },
        "inference_params": first_batch["inference_params"],
        "results": all_results,
    }

    # Preserve retrieval metadata if present
    if "retrieval" in first_batch.get("metadata", {}):
        aggregated["metadata"]["retrieval"] = first_batch["metadata"]["retrieval"]

    return aggregated


def infer_evaluation_filename(batch_files: List[str], timestamp: str) -> str:
    """Infer evaluation output filename from batch file names.

    Extracts common parts (split, seed, batch_size) and includes first/last question numbers.
    Format: {timestamp}_{split}_seed={seed}_bs={batch_size}_q{first_q}-{last_q}_evaluation.json
    """
    if not batch_files:
        raise ValueError("Cannot infer filename from empty file list")

    # Get the first filename to extract the pattern
    first_filename = os.path.basename(batch_files[0])

    # Pattern: {split}_seed={seed}_bn={batch_number}_bs={batch_size}.json
    # Extract: split, seed, batch_size
    pattern = r"^(.+?)_seed=(\d+)_bn=\d+_bs=(\d+)\.json$"
    match = re.match(pattern, first_filename)

    if not match:
        # Fallback: try to extract what we can and create a generic name
        base_name = os.path.splitext(first_filename)[0]
        # Remove bn=* part if present
        base_name = re.sub(r"_bn=\d+", "", base_name)

        # Try to extract batch info and calculate question range
        batch_numbers = []
        batch_size = None
        for batch_file in batch_files:
            bn, bs = extract_batch_info(os.path.basename(batch_file))
            if bn is not None:
                batch_numbers.append(bn)
            if bs is not None and batch_size is None:
                batch_size = bs

        if batch_numbers and batch_size:
            first_q, last_q = calculate_question_range(batch_numbers, batch_size)
            return f"{timestamp}_{base_name}_q{first_q}-{last_q}_evaluation.json"
        return f"{timestamp}_{base_name}_evaluation.json"

    split, seed, batch_size_str = match.groups()
    batch_size = int(batch_size_str)

    # Extract batch numbers from all files
    batch_numbers = []
    for batch_file in batch_files:
        bn, _ = extract_batch_info(os.path.basename(batch_file))
        if bn is not None:
            batch_numbers.append(bn)

    if not batch_numbers:
        return f"{timestamp}_{split}_seed={seed}_bs={batch_size}_evaluation.json"

    # Calculate question range
    first_q, last_q = calculate_question_range(batch_numbers, batch_size)

    return f"{timestamp}_{split}_seed={seed}_bs={batch_size}_q{first_q}-{last_q}_evaluation.json"


def build_pattern_from_args(args: argparse.Namespace) -> str:
    """Build glob pattern from directory and parameters."""
    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")

    # Build pattern: {split}_seed={seed}_bn=*_bs={batch_size}.json
    pattern = f"{args.split}_seed={args.seed}_bn=*_bs={args.batch_size}.json"
    return os.path.join(args.input_dir, pattern)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate preds JSON and write metrics to results/eval"
    )

    # Three modes: single file, pattern-based, or directory-based
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--preds", type=str, help="Path to a single preds JSON file")
    input_group.add_argument(
        "--pattern",
        type=str,
        help="Glob pattern to match batch files (e.g., 'preds/icl/hotpotqa_distractor/gpt-4/validation_seed=0_bn=*_bs=2.json')",
    )
    input_group.add_argument("--input-dir", type=str, help="Input directory containing batch files")

    # Required for directory-based mode
    parser.add_argument("--split", type=str, help="Dataset split (required with --input-dir)")
    parser.add_argument("--seed", type=int, help="Random seed (required with --input-dir)")
    parser.add_argument("--batch-size", type=int, help="Batch size (required with --input-dir)")

    parser.add_argument(
        "--outdir", default=os.path.join(REPO_ROOT, "results", "eval"), help="Output directory"
    )
    # Optional overrides
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--setting", default=None)
    parser.add_argument("--source", default="hf")
    parser.add_argument("--agent", default=None)
    parser.add_argument("--llm", default=None)

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [evaluate] %(message)s",
    )

    args = parser.parse_args()

    # Determine which files to evaluate
    tmp_file_path = None
    original_preds_path = None  # Store the original input for metadata
    if args.preds:
        # Single file mode
        preds_path = args.preds
        batch_files = [preds_path]
        original_preds_path = args.preds
    elif args.pattern:
        # Pattern-based mode
        batch_files = sorted(glob.glob(args.pattern))
        if not batch_files:
            raise ValueError(f"No files found matching pattern: {args.pattern}")
        logging.info("Found %d batch files to evaluate", len(batch_files))
        original_preds_path = args.pattern  # Store pattern for metadata
        # Aggregate files into a temporary file
        aggregated_data = aggregate_batch_files(batch_files)
        tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(aggregated_data, tmp_file, ensure_ascii=False, indent=2)
        tmp_file.close()
        preds_path = tmp_file.name
        tmp_file_path = tmp_file.name
    else:
        # Directory-based mode
        if not args.split or args.seed is None or not args.batch_size:
            parser.error("--split, --seed, and --batch-size are required when using --input-dir")
        pattern = build_pattern_from_args(args)
        batch_files = sorted(glob.glob(pattern))
        if not batch_files:
            raise ValueError(f"No files found matching pattern: {pattern}")
        logging.info("Found %d batch files to evaluate", len(batch_files))
        original_preds_path = pattern  # Store pattern for metadata
        # Aggregate files into a temporary file
        aggregated_data = aggregate_batch_files(batch_files)
        tmp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(aggregated_data, tmp_file, ensure_ascii=False, indent=2)
        tmp_file.close()
        preds_path = tmp_file.name
        tmp_file_path = tmp_file.name

    # Extract dataset and setting from pred file metadata if not provided via args
    if batch_files:
        pred_metadata = extract_metadata_from_pred_file(batch_files[0])
        if args.dataset is None and "dataset" in pred_metadata:
            args.dataset = pred_metadata["dataset"]
        if args.setting is None and "setting" in pred_metadata:
            args.setting = pred_metadata["setting"]

    # Evaluate
    results = evaluate_file(
        preds_path,
        dataset=args.dataset,
        setting=args.setting,
        split=args.split,
        source=args.source,
    )

    # Build timestamp MM-DD-HH
    timestamp = datetime.now().strftime("%m-%d-%H")

    # Override meta fields if provided (or extracted from pred file)
    meta = results["meta"]
    if args.dataset is not None:
        meta["dataset"] = args.dataset
    if args.agent is not None:
        meta["agent"] = args.agent
    if args.llm is not None:
        meta["llm"] = args.llm
    if args.setting is not None:
        meta["setting"] = args.setting
    if args.split is not None:
        meta["split"] = args.split
    if args.seed is not None:
        meta["seed"] = args.seed
    # Override preds_path to show the original input (pattern or file) instead of temp file
    if original_preds_path:
        meta["preds_path"] = original_preds_path
    # Save timestamp in the JSON payload
    meta["timestamp"] = timestamp

    # Determine output filename
    if args.preds:
        # Single file: use old naming convention
        filename = build_output_filename(
            dataset=meta.get("dataset", "unknown"),
            agent=meta.get("agent", "unknown"),
            llm=meta.get("llm", "unknown"),
            bn=int(meta.get("bn", -1)),
            bs=int(meta.get("bs", -1)),
            timestamp=timestamp,
        )
    else:
        # Multiple files: use new naming convention with question ranges
        filename = infer_evaluation_filename(batch_files, timestamp)

    outpath = save_results(results, args.outdir, filename)
    logging.info(f"Evaluation results saved to: {outpath}")
    print(outpath)

    # Clean up temporary file if created
    if tmp_file_path and os.path.exists(tmp_file_path):
        os.unlink(tmp_file_path)


if __name__ == "__main__":
    main()
