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
from datetime import datetime

# Fix OpenMP conflict when multiple libraries link to different OpenMP runtimes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from constants import REPO_ROOT

import torch

from agent import get_agent, Agent
from llm import get_llm
from data import get_dataset
from data.hotpotqa import load_hotpotqa_rag_corpus
from data.musique import load_musique_rag_corpus, write_musique_rag_corpus_jsonl

DEFAULT_FULLWIKI_CORPUS_PATH = "/share/j_sun/lmlm_multihop/datasets/hotpot_dev_fullwiki_v1.json"



from eval.evaluate import (
    evaluate_file,
    build_output_filename,
    save_results,
)

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


def _normalize_title(title: Any) -> str:
    return str(title or "").strip().lower()


def _extract_retrieved_title(doc: Any) -> str:
    if isinstance(doc, dict):
        return _normalize_title(doc.get("title", ""))
    return ""


def _compute_retrieval_stats(
    evidence_docs: List[Any],
    supporting_facts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    supporting_titles = {
        _normalize_title(item.get("title", ""))
        for item in (supporting_facts or [])
        if isinstance(item, dict)
    }
    supporting_titles = {t for t in supporting_titles if t}
    retrieved_titles = {
        _extract_retrieved_title(doc) for doc in (evidence_docs or [])
    }
    retrieved_titles = {t for t in retrieved_titles if t}

    overlap = supporting_titles.intersection(retrieved_titles)
    gold_total = len(supporting_titles)
    retrieved_total = len(retrieved_titles)
    overlap_count = len(overlap)

    return {
        "gold_total": gold_total,
        "retrieved_total": retrieved_total,
        "overlap": overlap_count,
        "precision": overlap_count / retrieved_total if retrieved_total else 0.0,
        "recall": overlap_count / gold_total if gold_total else 0.0,
    }


def process_single_batch(
    args: argparse.Namespace,
    batch_number: int,
    total_examples: int,
    full_dataset,
    agent: Agent,
    existing_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Process a single batch and return results dict."""
    logger = logging.getLogger("run_agent")

    # Calculate batch indices starting from start_index
    start_idx = args.start_index + (batch_number - 1) * args.batch_size
    end_idx = min(start_idx + args.batch_size, total_examples)

    if start_idx >= total_examples:
        return {}

    # Select batch slice
    ds = full_dataset.select(range(start_idx, end_idx))

    # Check if this batch is already complete (for resume)
    if args.resume and len(ds) > 0:
        first_qid = str(ds[0].get("id") or ds[0].get("_id"))
        if first_qid in existing_results:
            logger.info(f"Skipping batch {batch_number} (already processed)")
            return {}

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

    # Collect queries and info for batch
    queries = []
    examples_metadata = []

    for ex in ds:
        qid = ex.get("id") or ex.get("_id")
        question = ex["question"]
        contexts = ex.get("contexts") or []

        queries.append(question)
        examples_metadata.append({
            "qid": qid,
            "question": question,
            "answers": ex["answers"],
            "supporting_facts": ex["supporting_facts"],
            "contexts": contexts,
        })

    logger.info(f"Processing batch of {len(queries)} queries")

    if args.method in ("rag", "icl"):
        agent.reset(contexts)  # type: ignore

    answers, traces = agent.run(
        queries,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # Format results
    for idx, metadata in tqdm(enumerate(examples_metadata), total=len(examples_metadata), desc="Processing queries"):

        answer = answers[idx] if idx < len(answers) else None
        trace = traces[idx] if idx < len(traces) else None

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
                "golden_triplets" : step.golden_triplets,
            }
            for step in (trace or [])
        ]

        results[str(metadata["qid"])] = {
            "pred": answer,
            "gold_answer": metadata["answers"],
            "gold_evidence": metadata["supporting_facts"],
            "question": metadata["question"],
            "trace": serialized_trace,
        }
        if args.method == "rag":
            results[str(qid)]["retrieval"] = _compute_retrieval_stats(
                evidence_docs=evidence_docs,
                supporting_facts=ex.get("supporting_facts") or [],
            )
        if args.method == "rag" and args.debug_evidence:
            results[str(metadata["qid"])]["evidence"] = evidence_docs

    logger.info("Generated %d predictions for batch %d", len(results), batch_number)

    # Force garbage collection and clear GPU cache to prevent memory fragmentation
    gc.collect()
    if torch is not None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return results


def save_results_to_file(
    all_results: Dict[str, Dict[str, Any]],
    save_path: str,
    args: argparse.Namespace,
) -> None:
    """Save results to JSON file."""
    logger = logging.getLogger("run_agent")

    # Build final output with metadata
    output = {
        "metadata": {
            "model-path": args.model_path,
            "database-path": args.database_path,
            "model": args.model,
            "dataset": args.dataset,
            "setting": args.setting,
            "split": args.split,
            "batch_size": args.batch_size,
            "total_examples": len(all_results),
            "type": args.method,
            "seed": args.seed if args.seed is not None else None,
        },
        "inference_params": {
            "seed": args.seed,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        "results": all_results,
    }
    # Add retrieval metadata for RAG
    if args.method == "rag":
        output["metadata"]["retrieval"] = {
            "backend": args.retrieval,
            "scope": args.setting,
            "k": args.rag_k,
        }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    logger.info("Saved %d predictions to %s", len(all_results), save_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent over a dataset and save predictions.")
    parser.add_argument("--dataset", choices=["hotpotqa", "musique", "2wiki"], help="Dataset name")
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
        action="store_true"
    )
    parser.add_argument(
        "--top-k",
        default=4,
        help="Maximum number of results to retrieve from database",
    )
    parser.add_argument(
        "--return-triplets",
        default=False,
        help="Whether to return entire triplets (as opposed to only values)",
        action="store_true"
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
        help=(
            "Optional path to a HotpotQA/MuSiQue JSON or JSONL file to build a global RAG corpus. "
            "For MuSiQue, you can pass 'hf:<split>' (e.g., hf:train) to build and cache a JSONL."
        ),
    )

    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max output tokens")
    parser.add_argument(
        "--max-steps", type=int, default=5, help="Max reasoning steps for the Agent"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Dataset index to start from (0-based). Useful for parallelization.",
    )
    parser.add_argument(
        "--total-count",
        type=int,
        default=1000,
        help="Total number of examples to process from start-index. Default is 1000.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save results every N batches. None saves once at the end",
    )
    parser.add_argument(
        "--use-inverses",
        default=False,
        help="Whether to allow inverse lookups, of the form (value, relationship)",
        action="store_true"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results file, skipping already processed examples.",
    )
    parser.add_argument(
        "--output-dir", default=None, help="Base output directory (defaults to <repo>/preds)"
    )
    parser.add_argument(
        "--save-version", default=None, help="Save version (defaults to "")"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate the predictions",
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
    if args.method == "rag" and args.dataset == "musique" and not args.rag_corpus_path:
        args.rag_corpus_path = f"hf:{args.split}"
        logger.info(
            "MuSiQue RAG requires a global corpus; defaulting to %s",
            args.rag_corpus_path,
        )

    rag_corpus = None
    rag_corpus_path = args.rag_corpus_path
    rag_scope = None
    if (
        args.method == "rag"
        and args.dataset == "musique"
        and isinstance(rag_corpus_path, str)
        and rag_corpus_path.startswith("hf:")
    ):
        hf_split = rag_corpus_path.split(":", 1)[1].strip() or "train"
        cache_dir = os.path.join(REPO_ROOT, "preds", "rag_corpus")
        rag_corpus_path = os.path.join(cache_dir, f"musique_{hf_split}.jsonl")
        logger.info(
            "Building MuSiQue RAG corpus from HF split=%s and saving to %s",
            hf_split,
            rag_corpus_path,
        )
        count = write_musique_rag_corpus_jsonl(
            path=rag_corpus_path,
            split=hf_split,
            limit=None,
            seed=args.seed,
        )
        logger.info("Wrote %d unique RAG paragraphs to %s", count, rag_corpus_path)
        rag_scope = f"hf_{hf_split}"

    if args.method == "rag" and rag_corpus_path:
        logger.info("Loading RAG corpus from %s", rag_corpus_path)
        if args.dataset == "hotpotqa":
            rag_corpus = load_hotpotqa_rag_corpus(rag_corpus_path)
            logger.info(f"Loaded {len(rag_corpus)} unique RAG paragraphs from hotpotqa")
        elif args.dataset == "musique":
            rag_corpus = load_musique_rag_corpus(rag_corpus_path)
            logger.info(f"Loaded {len(rag_corpus)} unique RAG paragraphs from musique")
        else:
            rag_corpus = []
        
        # if rag corpus loading fails
        if not rag_corpus:
            logger.warning(
                "RAG corpus is empty after loading %s (check format and content).",
                rag_corpus_path,
            )
            raise RuntimeError(
                "RAG requires a non-empty global corpus; "
                "please verify --rag-corpus-path or HF cache."
            )

    if rag_scope is None:
        if args.method == "rag" and rag_corpus_path:
            rag_scope = _infer_rag_scope(rag_corpus_path)
        else:
            rag_scope = args.setting
    args.rag_scope = rag_scope

    print("split is :", args.split)

    # Load full dataset once (with seed for deterministic shuffling)
    full_dataset = get_dataset(name = args.dataset, setting = args.setting, split =  args.split, seed=args.seed)
    total_dataset_size = len(full_dataset)

    # Validate start_index
    if args.start_index >= total_dataset_size:
        logger.warning(f"Start index {args.start_index} is at or beyond dataset size {total_dataset_size}")
        return

    # Calculate how many examples to process (total_count is NUMBER of examples from start_index)
    examples_to_process = min(args.total_count, total_dataset_size - args.start_index)

    # TODO: how to make the training and eval use the same split function (e.g. create_train_val_splits)?
    # Calculate the exclusive end index
    end_index = args.start_index + examples_to_process

    logger.info(f"Dataset size: {total_dataset_size}, Processing {examples_to_process} examples from index {args.start_index} to {end_index}")

    # Prepare output location
    base_output_dir = args.output_dir or os.path.join(REPO_ROOT, "preds")
    if args.method == 'lmlm':
        model_name = args.model_path.split('/')[-1] if "checkpoint" not in args.model_path else args.model_path.split('/')[-2]+"-ckpt"+args.model_path.split('/')[-1].split("checkpoint-")[-1]
        output_dir = os.path.join(base_output_dir, args.method, args.dataset, model_name)
        use_inv_str = "_inv" if args.use_inverses else ""
        save_postfix = f"{args.dataset}_{args.split}_{model_name}_n{examples_to_process}_i{args.start_index}{use_inv_str}.json"
        save_path = os.path.join(output_dir, f"generations{args.save_version}", f"eval_{save_postfix}")
        save_results_path = os.path.join(output_dir, f"results{args.save_version}", f"results_{save_postfix}")
    else:
        model_name = args.model or "unknown-model"
        save_path = os.path.join(base_output_dir, args.method, f"{args.dataset}_{args.setting}", model_name, "generations")
        save_results_path = os.path.join(base_output_dir, args.method, f"{args.dataset}_{args.setting}", model_name, "results")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_results_path), exist_ok=True)

    # Load existing results if resuming
    existing_results = {}
    if args.resume and os.path.exists(save_path):
        try:
            with open(save_path, "r") as f:
                existing_data = json.load(f)
                existing_results = existing_data.get("results", {})
                logger.info(f"Resuming from {save_path} with {len(existing_results)} existing results")
        except Exception as e:
            logger.warning(f"Failed to load existing results from {save_path}: {e}")

    # Check if we already have all the results we need (and not resuming)
    if args.resume and os.path.exists(save_path):
        try:
            with open(save_path, "r") as f:
                existing_data = json.load(f)

            if (len(existing_data["results"]) >= examples_to_process and
                existing_data["metadata"]["model-path"] == args.model_path):
                logger.info(f"Generations already complete at {save_path} ({len(existing_data['results'])} results). Evaluating...")
                if args.eval:
                    # Evaluate
                    results = evaluate_file(
                        save_path,
                        dataset=args.dataset,
                        setting=args.setting,
                        split=args.split,
                        source='hf',
                    )
                    logger.info(f"Evaluation results: {json.dumps(results, indent=2)}")

                    outpath = save_results(results, "./", save_results_path)
                    logger.info(f"Evaluation results saved to: {outpath}")
                return
        except Exception as e:
            logger.warning(f"Failed to read existing results from {save_path}: {e}")

    # Calculate number of batches needed
    batches_to_process = (examples_to_process + args.batch_size - 1) // args.batch_size

    logger.info(
        "Processing %d examples in %d batches (starting from index %d): dataset=%s setting=%s split=%s method=%s model=%s batch_size=%d",
        examples_to_process,
        batches_to_process,
        args.start_index,
        args.dataset,
        args.setting,
        args.split,
        args.method,
        args.model,
        args.batch_size,
    )

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
        "max_steps": args.max_steps,
        "model_path": args.model_path,
        "database_path": args.database_path,
        "return_triplets" : args.return_triplets,
        "use_inverses" : args.use_inverses
    }

    # Get agent instance using factory function
    agent: Agent = get_agent(method=args.method, agent_kwargs=agent_kwargs)

    # Process batches with progress tracking
    # Start with existing results if resuming
    all_results = existing_results.copy() if existing_results else {}
    successful_batches = 0
    failed_batches = 0

    with tqdm(total=batches_to_process, desc="Processing batches", unit="batch") as pbar:
        for batch_num in range(1, batches_to_process + 1):
            try:
                batch_results = process_single_batch(
                    args, batch_num, end_index, full_dataset, agent, all_results
                )
                all_results.update(batch_results)
                if batch_results:  # Only count as successful if we actually processed
                    successful_batches += 1

                # Save based on save_every flag
                if args.save_every and successful_batches % args.save_every == 0:
                    save_results_to_file(all_results, save_path, args)

                pbar.update(1)
            except Exception as e:
                failed_batches += 1
                logger.error("Error processing batch %d: %s", batch_num, e, exc_info=True)
                pbar.update(1)
                raise
                
    logger.info(
        "Completed %d/%d batches successfully (%d failed)",
        successful_batches,
        batches_to_process,
        failed_batches,
    )

    # Save final results (either first time if save_every=-1, or final update)
    save_results_to_file(all_results, save_path, args)

    if args.eval:
        # Evaluate
        results = evaluate_file(
            save_path,
            dataset=args.dataset,
            setting=args.setting,
            split=args.split,
            source='hf',
        )
        logging.info(f"Evaluation results: {json.dumps(results, indent=2)}")


        outpath = save_results(results, "./", save_results_path)
        logging.info(f"Evaluation results saved to: {outpath}")



if __name__ == "__main__":
    main()