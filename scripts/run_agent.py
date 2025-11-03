#!/usr/bin/env python3
"""CLI to run the agent over a dataset and save predictions under preds/.

- Instantiates LLM and Agent
- Loads HotpotQA via Hugging Face datasets
- Runs the agent over questions (optionally limited)
- Saves predictions as JSON at preds/{method}/{dataset}_{setting}_{split}_bn={bn}.json

The JSON format is an object keyed by example id (pandas orient="index") where each
row contains:
  - pred: string prediction
  - metadata: { model, split, batch_size, batch_number, type }
  - inference_params: { seed, temperature, max_tokens }
This matches the expectations of evals/utils/_get_preds.
"""

import argparse
import json
import os
import random
import sys
import logging
from typing import Dict, Any


# Ensure imports work when running directly from repo
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets import load_dataset  # type: ignore

from src.agent.agent import Agent
from src.llm.openai import OpenAILLM
from src.llm.base import LLM
from src.llm import get_llm
from src.data import get_dataset


def build_query(question: str) -> str:
    """Instruction to ensure the Agent emits a FINAL_ANSWER the parser recognizes."""
    instruction = (
        "Provide only the final answer prefixed by 'FINAL_ANSWER:' with no extra text."
    )
    return f"{instruction}\n{question}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent over a dataset and save predictions.")
    parser.add_argument("--dataset", default="hotpotqa", choices=["hotpotqa"], help="Dataset name")
    parser.add_argument("--setting", default="distractor", choices=["distractor", "fullwiki"], help="Dataset setting")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "validation", "test"], help="Dataset split")
    parser.add_argument("--method", default="icl", choices=["db", "rag", "icl"], help="Agent method label (for output path)")
    parser.add_argument("--model", default="gpt-4", help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max output tokens")
    parser.add_argument("--max-steps", type=int, default=5, help="Max reasoning steps for the Agent")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--batch-number", type=int, default=1, help="Batch number index (1-based)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--output-dir", default=None, help="Base output directory (defaults to <repo>/preds)")
    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [run_agent] %(message)s",
    )
    logger = logging.getLogger("run_agent")

    random.seed(args.seed)

    # Instantiate LLM and Agent
    llm = get_llm(model_name=args.model)
    agent = Agent(llm=llm, max_steps=args.max_steps)

    # Load dataset
    ds = get_dataset(args.dataset, args.setting, args.split)

    logger.info(
        "Starting run: dataset=%s setting=%s split=%s method=%s model=%s batch_number=%d batch_size=%s",
        args.dataset,
        args.setting,
        args.split,
        args.method,
        args.model,
        args.batch_number,
        str(args.batch_size),
    )

    total = len(ds)

    # Batch slicing (1-based batch_number)
    start_idx = (args.batch_number - 1) * args.batch_size
    end_idx = min(start_idx + args.batch_size, len(ds))
    if start_idx < len(ds):
        ds = ds.select(range(start_idx, end_idx))
        logger.info("Selected batch %d: indices [%d, %d) => %d examples", args.batch_number, start_idx, end_idx, len(ds))
    else:
        ds = ds.select([])
        logger.warning("Batch %d is out of range (start_idx=%d >= %d). No examples to process.", args.batch_number, start_idx, total)

    # Prepare output location
    base_output_dir = args.output_dir or os.path.join(REPO_ROOT, "preds")
    method_dir = os.path.join(base_output_dir, args.method)
    os.makedirs(method_dir, exist_ok=True)
    filename = f"{args.dataset}_{args.setting}_{args.split}_bn={args.batch_number}_bs={args.batch_size}.json"
    output_path = os.path.join(method_dir, filename)

    # Run predictions
    predictions: Dict[str, Dict[str, Any]] = {}
    batch_size = len(ds)
    for ex in ds:
        qid = ex.get("id") or ex.get("_id")
        question = ex["question"]
        query = build_query(question)

        answer, trace = agent.run(
            query,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        logging.info("Answer: %s", answer)
        logging.info("Trace: %s", trace)

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
        
        predictions[str(qid)] = {
            "pred": answer,
            "gold_answer": ex["answers"],
            "gold_evidence": ex["supporting_facts"],
            "question": question,
            "trace": serialized_trace,
            "metadata": {
                "model": args.model,
                "split": args.split,
                "batch_size": batch_size,
                "batch_number": args.batch_number,
                "type": args.method,
            },
            "inference_params": {
                "seed": args.seed,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            },
        }
    
    logger.info("Generated %d predictions", len(predictions))
    logger.debug("Predictions payload: %s", json.dumps(predictions, ensure_ascii=False)[:2000])
    # Save JSON in pandas orient="index" compatible layout
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)

    logger.info("Saved %d predictions to %s", len(predictions), output_path)


if __name__ == "__main__":
    main()


