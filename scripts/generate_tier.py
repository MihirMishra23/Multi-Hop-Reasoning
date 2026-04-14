"""
Generate dataset of prompts with different difficulty levels.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from tqdm import tqdm
from constants import REPO_ROOT

SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

if TYPE_CHECKING:
    from agent import Agent


def _score_prediction(pred: str, gold_answers: List[str]) -> Tuple[float, int]:
    """Return (max_f1, em) across gold answers."""
    from eval.metrics import exact_match_score, f1_score

    if not gold_answers:
        return 0.0, 0
    max_f1 = 0.0
    em = 0
    for gold in gold_answers:
        f1, _, _ = f1_score(pred, gold)
        if f1 > max_f1:
            max_f1 = f1
        if exact_match_score(pred, gold):
            em = 1
    return max_f1, em


def _validate_thresholds(answer_threshold: float) -> None:
    if not (0.0 <= answer_threshold <= 1.0):
        raise ValueError("Answer threshold must satisfy 0 <= answer_threshold <= 1.")


def _init_agent(args: argparse.Namespace) -> "Agent":
    from agent import get_agent
    from llm import get_llm

    llm = None
    if args.method not in ("lmlm", "two_phase"):
        if not args.model:
            raise ValueError("You must set --model for non-lmlm methods.")
        llm = get_llm(model_name=args.model)

    agent_kwargs: Dict[str, Any] = {
        "llm": llm,
        "model": args.model,
        "dataset": args.dataset,
        "setting": args.setting,
        "retrieval": args.retrieval,
        "rag_k": args.rag_k,
        "max_steps": args.max_steps,
        "model_path": args.model_path,
        "database_path": args.database_path,
        "return_triplets": args.return_triplets,
        "use_inverses": args.use_inverses,
        "top_k": args.top_k,
        "similarity_threshold": args.similarity_threshold,
    }
    return get_agent(method=args.method, agent_kwargs=agent_kwargs)


def _build_output_path(args: argparse.Namespace, num_samples: int) -> str:
    base_dir = args.output_dir or os.path.join("output", "tiers")
    os.makedirs(base_dir, exist_ok=True)
    save_version = f"_{args.save_version}" if args.save_version else ""
    filename = (
        f"{args.dataset}_{args.split}_{args.method}_n{num_samples}_i{args.start_index}"
        f"{save_version}.json"
    )
    return os.path.join(base_dir, filename)


def _serialize_trace(steps) -> List[Dict[str, Any]]:
    return [
        {
            "prompt": step.prompt,
            "answer": step.answer,
            "action": step.action,
            "error": step.error,
            "tool_name": step.tool_name,
            "tool_args": step.tool_args,
            "golden_triplets": step.golden_triplets,
        }
        for step in (steps or [])
    ]


def _save_checkpoint(
    output_path: str,
    args: argparse.Namespace,
    results: Dict[str, Any],
    num_samples: int,
    completed_rollouts: int,
) -> None:
    logger = logging.getLogger("generate_tier")
    score_counts = {str(i): 0 for i in range(args.num_rollouts + 1)}
    for item in results.values():
        score = int(item.get("successes", 0))
        score_counts[str(min(score, args.num_rollouts))] = score_counts.get(str(min(score, args.num_rollouts)), 0) + 1

    output = {
        "metadata": {
            "dataset": args.dataset,
            "setting": args.setting,
            "split": args.split,
            "method": args.method,
            "model": args.model,
            "model_path": args.model_path,
            "database_path": args.database_path,
            "num_samples": num_samples,
            "total_count": args.total_count,
            "start_index": args.start_index,
            "seed": args.seed,
            "num_rollouts": args.num_rollouts,
            "completed_rollouts": completed_rollouts,
            "thresholds": {"answer": args.answer_threshold},
        },
        "score_counts": score_counts,
        "results": results,
    }
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_path)
    logger.info("Checkpoint saved → %s  (%d rollouts complete)", output_path, completed_rollouts)


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("generate_tier")

    _validate_thresholds(args.answer_threshold)

    from data import get_dataset

    dataset = get_dataset(
        name=args.dataset,
        setting=args.setting,
        split=args.split,
        seed=args.seed,
    )

    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if args.start_index >= len(dataset):
        raise ValueError("Start index is beyond dataset size.")

    total_count = args.total_count
    num_samples = min(total_count, len(dataset) - args.start_index)
    dataset = dataset.select(range(args.start_index, args.start_index + num_samples))

    logger.info(
        "Dataset: %s  split=%s  start=%d  num_samples=%d  rollouts=%d  method=%s",
        args.dataset, args.split, args.start_index, num_samples, args.num_rollouts, args.method,
    )

    if args.method in ("icl", "rag") and args.batch_size != 1:
        raise ValueError("ICL/RAG agents require --batch-size 1.")

    # Determine output path early so we can checkpoint to it
    output_path = _build_output_path(args, num_samples)
    logger.info("Output path: %s", output_path)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    results: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, encoding="utf-8") as f:
                ckpt = json.load(f)
            results = ckpt.get("results", {})
            # Recompute successes from rollouts for consistency
            for item in results.values():
                item["successes"] = sum(1 for r in item.get("rollouts", []) if r.get("correct", False))
            ckpt_rollouts = ckpt.get("metadata", {}).get("completed_rollouts", 0)
            logger.info(
                "Loaded checkpoint: %d results, %d rollouts previously completed",
                len(results), ckpt_rollouts,
            )
        except Exception as exc:
            logger.warning("Could not load checkpoint (%s) — starting fresh", exc)
            results = {}

    agent = _init_agent(args)

    if args.method in ("lmlm", "two_phase"):
        qids = [str(ex.get("id") or ex.get("_id") or idx) for idx, ex in enumerate(dataset)]
        queries = [ex["question"] for ex in dataset]
        contexts_list = None
        if args.method == "two_phase":
            contexts_list = [ex.get("golden_contexts") or ex.get("contexts") or [] for ex in dataset]

        # Initialise any missing entries (new run or fresh start)
        for idx, ex in enumerate(dataset):
            qid = qids[idx]
            if qid not in results:
                results[qid] = {
                    "question": ex.get("question"),
                    "answers": ex.get("answers"),
                    "rollouts": [],
                    "successes": 0,
                    "success_rate": 0.0,
                    "score": 0,
                }

        # Process in huge chunks: each chunk finishes all rollouts before moving on.
        n = len(queries)
        chunk_size = args.chunk_size if args.chunk_size > 0 else n
        num_chunks = (n + chunk_size - 1) // chunk_size

        for chunk_idx, chunk_start in enumerate(range(0, n, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, n)
            chunk_qids = qids[chunk_start:chunk_end]
            chunk_queries = queries[chunk_start:chunk_end]
            chunk_contexts = contexts_list[chunk_start:chunk_end] if contexts_list else None
            chunk_n = chunk_end - chunk_start

            # How many rollouts are fully done for this chunk?
            chunk_completed = min(len(results[qid]["rollouts"]) for qid in chunk_qids)

            if chunk_completed == args.num_rollouts:
                logger.info(
                    "── Chunk %d/%d [%d:%d] already complete, skipping ──",
                    chunk_idx + 1, num_chunks, chunk_start, chunk_end,
                )
                continue

            if chunk_completed > 0:
                logger.info(
                    "── Chunk %d/%d [%d:%d] resuming from rollout %d ──",
                    chunk_idx + 1, num_chunks, chunk_start, chunk_end, chunk_completed,
                )

            chunk_t0 = time.time()
            for rollout_idx in range(chunk_completed, args.num_rollouts):
                rollout_t0 = time.time()
                logger.info(
                    "── Chunk %d/%d [%d:%d]  Rollout %d/%d starting  "
                    "(batch_size=%d, chunk_n=%d) ──",
                    chunk_idx + 1, num_chunks, chunk_start, chunk_end,
                    rollout_idx + 1, args.num_rollouts, args.batch_size, chunk_n,
                )

                all_answers: List[Any] = []
                all_traces: List[Any] = []
                num_batches = (chunk_n + args.batch_size - 1) // args.batch_size
                for batch_idx, batch_start_local in enumerate(range(0, chunk_n, args.batch_size)):
                    batch_end_local = min(batch_start_local + args.batch_size, chunk_n)
                    batch_queries = chunk_queries[batch_start_local:batch_end_local]
                    run_kwargs: Dict[str, Any] = dict(
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                    if args.method == "two_phase":
                        run_kwargs["contexts"] = chunk_contexts[batch_start_local:batch_end_local]
                    b_answers, b_traces = agent.run(batch_queries, **run_kwargs)
                    all_answers.extend(b_answers)
                    all_traces.extend(b_traces)
                    if (batch_idx + 1) % 10 == 0 or batch_end_local == chunk_n:
                        elapsed = time.time() - rollout_t0
                        rate = batch_end_local / elapsed if elapsed > 0 else 0
                        eta = (chunk_n - batch_end_local) / rate if rate > 0 else float("inf")
                        logger.info(
                            "  rollout %d/%d — batch %d/%d  (%d/%d examples)  "
                            "%.1f ex/s  ETA %.0fs",
                            rollout_idx + 1, args.num_rollouts,
                            batch_idx + 1, num_batches,
                            batch_end_local, chunk_n, rate, eta,
                        )

                # Accumulate scores for this rollout
                rollout_successes = 0
                for i, qid in enumerate(chunk_qids):
                    pred = all_answers[i] or ""
                    f1, em = _score_prediction(pred, dataset[chunk_start + i].get("answers") or [])
                    correct = f1 >= args.answer_threshold
                    if correct:
                        results[qid]["successes"] += 1
                        rollout_successes += 1
                    results[qid]["rollouts"].append(
                        {
                            "rollout": rollout_idx,
                            "pred": pred,
                            "f1": f1,
                            "em": em,
                            "correct": correct,
                            "trace": _serialize_trace(all_traces[i]),
                        }
                    )

                rollout_elapsed = time.time() - rollout_t0
                chunk_elapsed = time.time() - chunk_t0
                done_in_chunk = rollout_idx + 1 - chunk_completed
                remaining_in_chunk = args.num_rollouts - rollout_idx - 1
                eta_chunk = (chunk_elapsed / done_in_chunk) * remaining_in_chunk if done_in_chunk > 0 else float("inf")
                logger.info(
                    "── Chunk %d/%d  Rollout %d/%d done in %.1fs  "
                    "rollout_acc=%.3f  chunk_ETA ~%.0fs ──",
                    chunk_idx + 1, num_chunks,
                    rollout_idx + 1, args.num_rollouts,
                    rollout_elapsed, rollout_successes / chunk_n, eta_chunk,
                )

                # Save checkpoint after every completed rollout within the chunk
                _save_checkpoint(output_path, args, results, num_samples, rollout_idx + 1)

            logger.info(
                "── Chunk %d/%d [%d:%d] fully complete ──",
                chunk_idx + 1, num_chunks, chunk_start, chunk_end,
            )

        # Finalise scores
        for qid in results:
            successes = results[qid]["successes"]
            results[qid]["success_rate"] = successes / float(args.num_rollouts)
            results[qid]["score"] = successes

    else:
        # icl / rag / db — sequential per-example
        for ex in tqdm(dataset, total=len(dataset), desc="Running agent"):
            qid = str(ex.get("id") or ex.get("_id") or "")
            question = ex.get("question")
            contexts = ex.get("contexts") or []
            results[qid] = {
                "question": question,
                "answers": ex.get("answers"),
                "rollouts": [],
                "successes": 0,
                "success_rate": 0.0,
                "score": 0,
            }

            for rollout_idx in range(args.num_rollouts):
                if args.method in ("icl", "rag"):
                    agent.reset(contexts)  # type: ignore
                answers, traces = agent.run(
                    [question],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                pred = answers[0] or ""
                f1, em = _score_prediction(pred, ex.get("answers") or [])
                correct = f1 >= args.answer_threshold
                if correct:
                    results[qid]["successes"] += 1
                results[qid]["rollouts"].append(
                    {
                        "rollout": rollout_idx,
                        "pred": pred,
                        "f1": f1,
                        "em": em,
                        "correct": correct,
                        "trace": _serialize_trace(traces[0]),
                    }
                )

            success_rate = results[qid]["successes"] / float(args.num_rollouts)
            results[qid]["success_rate"] = success_rate
            results[qid]["score"] = results[qid]["successes"]

    # Final save (also finalises score_counts)
    _save_checkpoint(output_path, args, results, num_samples, args.num_rollouts)
    logger.info("Done. Results saved to %s", output_path)
    if args.plot_path:
        score_counts = {str(i): 0 for i in range(args.num_rollouts + 1)}
        for item in results.values():
            score = int(item.get("score", 0))
            score_counts[str(score)] = score_counts.get(str(score), 0) + 1
        visualize_score_distribution(score_counts, args.num_rollouts, args.plot_path)


def visualize_score_distribution(
    score_counts: Dict[str, int],
    num_rollouts: int,
    output_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install it or omit --plot-path."
        ) from exc

    labels = [str(i) for i in range(num_rollouts + 1)]
    values = [score_counts.get(label, 0) for label in labels]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color="#4caf50")
    plt.title("Rollout Success Count Distribution")
    plt.xlabel("Successes (0..num_rollouts)")
    plt.ylabel("Count")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run agent over a dataset and classify question difficulty."
    )
    parser.add_argument("--dataset", choices=["hotpotqa", "musique", "2wiki"], required=True)
    parser.add_argument("--setting", default="distractor", choices=["distractor", "fullwiki"])
    parser.add_argument("--split", default="dev", choices=["train", "dev", "validation", "test"])
    parser.add_argument("--method", default="icl", choices=["db", "rag", "icl", "lmlm", "two_phase"])
    parser.add_argument("--model", default=None, help="LLM model name for icl/rag.")
    parser.add_argument("--model-path", default=None, help="Local model path for lmlm.")
    parser.add_argument("--database-path", default=None, help="Triplets DB path for lmlm.")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size (lmlm only).")
    parser.add_argument("--chunk-size", type=int, default=0, help="Huge-batch size: each chunk finishes all rollouts before the next chunk starts. 0 means process all samples as one chunk (default).")
    parser.add_argument("--total-count", type=int,required=True, help="Total number of examples to process from start-index (eval-compatible).")
    parser.add_argument("--num-rollouts", type=int, default=1, help="Number of rollouts per question.")
    parser.add_argument(
        "--answer-threshold",
        type=float,
        default=0.5,
        help="F1 threshold to count a rollout as correct.",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory (default: output/tiers).")
    parser.add_argument("--save-version", default=None, help="Optional suffix for output filename.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--retrieval", default="bm25", choices=["bm25"])
    parser.add_argument("--rag-k", type=int, default=4)
    parser.add_argument("--return-triplets", action="store_true")
    parser.add_argument("--use-inverses", action="store_true")
    parser.add_argument("--top-k", type=int, default=4, help="Number of triplets to retrieve per step (lmlm/two_phase).")
    parser.add_argument("--similarity-threshold", type=float, default=0.6, help="Retrieval similarity threshold (lmlm/two_phase).")
    parser.add_argument("--plot-path", default=None, help="Save score distribution plot.")
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
