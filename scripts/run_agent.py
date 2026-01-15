#!/usr/bin/env python3
"""CLI to run the agent over a dataset and save predictions under preds/.

- Instantiates LLM and Agent
- Loads HotpotQA via Hugging Face datasets
- Runs the agent over questions (optionally limited)
- Saves predictions with structure: preds/{type}/{dataset}_{setting}/{model}/{split}_seed={s}_bn={n}_bs={b}.json

The JSON format uses deduplicated metadata at the top level:
{
  "metadata": { model, split, batch_size, batch_number, type, seed, retrieval },
  "inference_params": { seed, temperature, max_tokens },
  "results": {
    "qid": { pred, gold_answer, gold_evidence, question, trace, evidence }
  }
}
"""

import argparse
import json
import os
import random
import sys
import logging
import gc
from typing import Dict, Any, List
from tqdm import tqdm

# Fix OpenMP conflict when multiple libraries link to different OpenMP runtimes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from constants import REPO_ROOT

import torch

from agent import get_agent, Agent
from llm import get_llm
from data import get_dataset
from data.hotpotqa import load_hotpotqa_rag_corpus

DEFAULT_FULLWIKI_CORPUS_PATH = "/share/j_sun/lmlm_multihop/datasets/hotpot_dev_fullwiki_v1.json"


def build_query(question: str) -> str:
    """Instruction to ensure the Agent emits a FINAL_ANSWER the parser recognizes."""
    instruction = "Provide only the final answer prefixed by 'FINAL_ANSWER:' with no extra text."
    return f"{instruction}\n{question}"


def _infer_rag_scope(rag_corpus_path: str) -> str:
    lower_name = os.path.basename(rag_corpus_path).lower()
    if "fullwiki" in lower_name:
        return "fullwiki"
    if "distractor" in lower_name:
        return "distractor"
    return "custom"


def process_single_batch(
    args: argparse.Namespace,
    batch_number: int,
    total_examples: int,
    output_dir: str,
    full_dataset,
    agent: Agent,
) -> bool:
    """Process a single batch and save immediately. Returns True if successful."""
    logger = logging.getLogger("run_agent")

    # Calculate batch indices
    start_idx = (batch_number - 1) * args.batch_size
    end_idx = min(start_idx + args.batch_size, total_examples)

    if start_idx >= total_examples:
        return False

    # Build output path
    filename = f"{args.split}_seed={args.seed}_bn={batch_number}_bs={args.batch_size}.json"
    output_path = os.path.join(output_dir, filename)

    # Check if already exists (for resume)
    if args.resume and os.path.exists(output_path):
        logger.info("Skipping batch %d (already exists at %s)", batch_number, output_path)
        return True

    # Select batch slice
    ds = full_dataset.select(range(start_idx, end_idx))
    logger.info(
        "Processing batch %d: indices [%d, %d) => %d examples",
        batch_number,
        start_idx,
        end_idx,
        len(ds),
    )

    # Run predictions
    results: Dict[str, Dict[str, Any]] = {}
    batch_size_actual = len(ds)

    count = 0
    for ex in ds:
        count += 1
        if (count %10 == 0):
            print(f"\n\ncount : {count} \n\n")
        qid = ex.get("id") or ex.get("_id")
        question = ex["question"]

        # Reset agent state for new question (with new contexts if applicable)
        contexts = ex.get("contexts") or []
        if args.method in ("rag", "icl"):
            agent.reset(contexts)  # type: ignore

        answer, trace = agent.run(
            question,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        # Extract evidence docs for RAG if needed
        evidence_docs = []
        if args.method == "rag":
            evidence_docs = getattr(agent, "_evidence_docs", [])
        logger.debug("Answer: %s", answer)
        logger.debug("Trace: %s", trace)

        # Fallback to last step text if FINAL_ANSWER not parsed
        if answer is None and trace:
            answer = (trace[-1].answer or "").strip()
        if answer is None:
            answer = ""

        # Serialize trace for JSON output
        serialized_trace = [
            {
                "prompt": step.prompt,
                "answer": step.answer,
                "action": step.action,
                "error": step.error,
                "tool_name": step.tool_name,
                "tool_args": step.tool_args,
            }
            for step in (trace or [])
        ]

        results[str(qid)] = {
            "pred": answer,
            "gold_answer": ex["answers"],
            "gold_evidence": ex["supporting_facts"],
            "question": question,
            "trace": serialized_trace,
        }
        if args.method == "rag" and args.debug_evidence:
            results[str(qid)]["evidence"] = evidence_docs

    # Build final output with deduplicated metadata
    output = {
        "metadata": {
            "model-path": args.model_path,
            "database-path": args.database_path,
            "model": args.model,
            "dataset": args.dataset,
            "setting": args.setting,
            "split": args.split,
            "batch_size": batch_size_actual,
            "batch_number": batch_number,
            "type": args.method,
            "seed": args.seed if args.seed is not None else None,
        },
        "inference_params": {
            "seed": args.seed,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        "results": results,
    }
    # Add retrieval metadata for RAG
    if args.method == "rag":
        output["metadata"]["retrieval"] = {
            "backend": args.retrieval,
            "scope": args.rag_scope,
            "k": args.rag_k,
        }

    logger.info("Generated %d predictions for batch %d", len(results), batch_number)
    logger.debug("Predictions payload: %s", json.dumps(output, ensure_ascii=False)[:2000])

    # Save JSON with deduplicated metadata immediately
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    logger.info("Saved %d predictions to %s", len(results), output_path)

    # Force garbage collection and clear GPU cache to prevent memory fragmentation
    gc.collect()
    if torch is not None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent over a dataset and save predictions.")
    parser.add_argument("--dataset", choices=["hotpotqa", "musique"], help="Dataset name")
    parser.add_argument(
        "--setting",
        default="distractor",
        choices=["distractor", "fullwiki"],
        help="Dataset setting",
    )
    parser.add_argument(
        "--split",
        default="dev",
        choices=["train", "dev", "validation", "test"],
        help="Dataset split",
    )
    parser.add_argument(
        "--method",
        default="icl",
        choices=["db", "rag", "icl", "lmlm"],
        help="Agent method label (for output path)",
    )
    parser.add_argument("--model-path", default=None, help="Local model path")
    parser.add_argument(
        "--database-path",
        default=None,
        help="Path to database of (entity, relation, value) triplets",
    )
    parser.add_argument(
        "--adaptive-k",
        default=False,
        help="Whether to use adaptive k for lmlm retreival",
    )
    # RAG-related flags
    parser.add_argument(
        "--retrieval", default="bm25", choices=["bm25"], help="Retrieval backend for --method rag"
    )
    parser.add_argument("--rag-k", type=int, default=4, help="Top-k documents to retrieve")
    parser.add_argument(
        "--debug-evidence",
        action="store_true",
        help="Include retrieved evidence in saved preds for debugging",
    )
    parser.add_argument(
        "--rag-corpus-path",
        default=None,
        help="Optional path to a HotpotQA JSON file to build a global RAG corpus.",
    )

    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max output tokens")
    parser.add_argument(
        "--max-steps", type=int, default=5, help="Max reasoning steps for the Agent"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--batch-number", type=int, default=1, help="Batch number index (1-based)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="Number of batches to process (default: 1). Use -1 to process all batches.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip batches that already exist (check by file existence)",
    )
    parser.add_argument(
        "--output-dir", default=None, help="Base output directory (defaults to <repo>/preds)"
    )
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [run_agent] %(message)s",
    )
    logger = logging.getLogger("run_agent")

    if (
        args.method == "rag"
        and args.dataset == "hotpotqa"
        and args.setting == "fullwiki"
        and not args.rag_corpus_path
    ):
        args.rag_corpus_path = DEFAULT_FULLWIKI_CORPUS_PATH

    rag_corpus = None
    if args.method == "rag" and args.dataset == "hotpotqa" and args.rag_corpus_path:
        logger.info("Loading RAG corpus from %s", args.rag_corpus_path)
        rag_corpus = load_hotpotqa_rag_corpus(args.rag_corpus_path)
        logger.info("Loaded %d unique RAG paragraphs", len(rag_corpus))
        if not rag_corpus:
            logger.warning(
                "RAG corpus is empty after loading %s (check format and content).",
                args.rag_corpus_path,
            )

    args.rag_scope = (
        _infer_rag_scope(args.rag_corpus_path)
        if args.method == "rag" and args.rag_corpus_path
        else args.setting
    )

    # Load full dataset once (with seed for deterministic shuffling)
    full_dataset = get_dataset(args.dataset, args.setting, args.split, seed=args.seed)
    total = len(full_dataset)

    # Prepare output location with structure: type/dataset_setting/model/split_seed{s}_bn{n}_bs{b}.json
    base_output_dir = args.output_dir or os.path.join(REPO_ROOT, "preds")

    # Build directory structure: type/dataset_setting/model/
    dataset_setting = f"{args.dataset}_{args.setting}"
    model_name = args.model or "unknown-model"
    output_dir = os.path.join(base_output_dir, args.method, dataset_setting, model_name)

    os.makedirs(output_dir, exist_ok=True)

    # Determine batch processing mode
    total_batches = (total + args.batch_size - 1) // args.batch_size

    # Process specified number of batches (or all if -1)
    if args.num_batches == -1:
        # Process all remaining batches
        num_batches_to_process = total_batches - args.batch_number + 1
        logger.info(
            "Processing all batches starting from batch %d: dataset=%s setting=%s split=%s method=%s model=%s batch_size=%d (total_batches=%d)",
            args.batch_number,
            args.dataset,
            args.setting,
            args.split,
            args.method,
            args.model,
            args.batch_size,
            total_batches,
        )
    else:
        # Process specified number of batches
        num_batches_to_process = min(args.num_batches, total_batches - args.batch_number + 1)
        logger.info(
            "Processing %d batch(es) starting from batch %d: dataset=%s setting=%s split=%s method=%s model=%s batch_size=%d (total_batches=%d)",
            num_batches_to_process,
            args.batch_number,
            args.dataset,
            args.setting,
            args.split,
            args.method,
            args.model,
            args.batch_size,
            total_batches,
        )

    start_batch = args.batch_number

    # Instantiate LLM and Agent once
    llm = None
    if args.method != "lmlm":
        llm = get_llm(model_name=args.model)

    # Build agent_kwargs dictionary
    agent_kwargs = {
        "llm": llm,
        "model": args.model,
        "dataset": args.dataset,
        "setting": args.setting,
        "retrieval": args.retrieval,
        "rag_k": args.rag_k,
        "rag_corpus": rag_corpus,
        "max_steps": args.max_steps,
        "model_path": args.model_path,
        "database_path": args.database_path,
    }

    # Get agent instance using factory function
    agent: Agent = get_agent(method=args.method, agent_kwargs=agent_kwargs)

    # Process batches with progress tracking
    successful_batches = 0
    failed_batches = 0

    with tqdm(total=num_batches_to_process, desc="Processing batches", unit="batch") as pbar:
        for batch_num in range(start_batch, start_batch + num_batches_to_process):
            try:
                success = process_single_batch(
                    args, batch_num, total, output_dir, full_dataset, agent
                )
                if success:
                    successful_batches += 1
                pbar.update(1)
            except Exception as e:
                failed_batches += 1
                logger.error("Error processing batch %d: %s", batch_num, e, exc_info=True)
                if not args.resume:
                    # Re-raise if not in resume mode to fail fast
                    raise
                pbar.update(1)

    logger.info(
        "Completed %d/%d batches successfully (%d failed)",
        successful_batches,
        num_batches_to_process,
        failed_batches,
    )


if __name__ == "__main__":
    main()
