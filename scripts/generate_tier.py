"""
Generate dataset of prompts with different difficulty levels.
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
    if args.method != "lmlm":
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

    agent = _init_agent(args)

    results: Dict[str, Dict[str, Any]] = {}

    if args.method in ("icl", "rag") and args.batch_size != 1:
        raise ValueError("ICL/RAG agents require --batch-size 1.")

    if args.method == "lmlm":
        queries = [ex["question"] for ex in dataset]
        for idx, ex in enumerate(dataset):
            qid = str(ex.get("id") or ex.get("_id") or idx)
            results[qid] = {
                "question": ex.get("question"),
                "answers": ex.get("answers"),
                "rollouts": [],
                "successes": 0,
                "success_rate": 0.0,
                "score": 0,
            }

        for rollout_idx in range(args.num_rollouts):
            answers, traces = agent.run(
                queries,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            for idx, ex in enumerate(dataset):
                qid = str(ex.get("id") or ex.get("_id") or idx)
                pred = answers[idx] or ""
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
                        "trace": [
                            {
                                "prompt": step.prompt,
                                "answer": step.answer,
                                "action": step.action,
                                "error": step.error,
                                "tool_name": step.tool_name,
                                "tool_args": step.tool_args,
                                "golden_triplets": step.golden_triplets,
                            }
                            for step in (traces[idx] or [])
                        ],
                    }
                )

        for qid in tqdm(results.keys(), total=len(results), desc="Scoring"):
            successes = results[qid]["successes"]
            success_rate = successes / float(args.num_rollouts)
            results[qid]["success_rate"] = success_rate
            results[qid]["score"] = successes
    else:
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
                        "trace": [
                            {
                                "prompt": step.prompt,
                                "answer": step.answer,
                                "action": step.action,
                                "error": step.error,
                                "tool_name": step.tool_name,
                                "tool_args": step.tool_args,
                                "golden_triplets": step.golden_triplets,
                            }
                            for step in (traces[0] or [])
                        ],
                    }
                )

            success_rate = results[qid]["successes"] / float(args.num_rollouts)
            results[qid]["success_rate"] = success_rate
            results[qid]["score"] = results[qid]["successes"]

    score_counts = {str(i): 0 for i in range(args.num_rollouts + 1)}
    for item in results.values():
        score = int(item.get("score", 0))
        score_counts[str(score)] = score_counts.get(str(score), 0) + 1

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
            "total_count": total_count,
            "start_index": args.start_index,
            "seed": args.seed,
            "num_rollouts": args.num_rollouts,
            "thresholds": {
                "answer": args.answer_threshold,
            },
        },
        "score_counts": score_counts,
        "results": results,
    }

    output_path = _build_output_path(args, num_samples)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("Saved tiered results to %s", output_path)
    if args.plot_path:
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
    parser.add_argument("--method", default="icl", choices=["db", "rag", "icl", "lmlm"])
    parser.add_argument("--model", default=None, help="LLM model name for icl/rag.")
    parser.add_argument("--model-path", default=None, help="Local model path for lmlm.")
    parser.add_argument("--database-path", default=None, help="Triplets DB path for lmlm.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (lmlm only).")
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
    parser.add_argument("--plot-path", default=None, help="Save score distribution plot.")
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
