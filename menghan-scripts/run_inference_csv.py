#!/usr/bin/env python3
"""Run two-phase inference from a checkpoint and save a CSV in the same format
as the wandb training logs.

Two modes (controlled by --concat-all-db):

  Default (per-example DB):
    Phase 1 runs per batch → each example gets its own DB → Phase 2 uses that DB.

  --concat-all-db (unified DB):
    Phase 1 runs once over the entire dataset via build_unified_db_from_dataset →
    one shared DB is built → Phase 2 for every example uses this unified DB.
    This is the recommended eval mode for comparing against training.

CSV columns (K=1, M=1 — advantages are NaN during inference):
    step, answer, phase1_prompt, phase2_prompt,
    phase1_completion_0, phase1_context_0, generated_db_0,
    phase1_advantage_0, db_size_threshold_0,
    phase2_completion_0_0, phase2_advantage_0_0, em_accuracy_0_0

Usage:
    cd /path/to/repo && python menghan-scripts/run_inference_csv.py \\
        --model-path /path/to/checkpoint-XXXX \\
        --dataset hotpotqa --split validation \\
        --total-count 200 --concat-all-db \\
        --output reward_hacking_evaluate/ckpt_XXXX.csv
"""

import argparse
import csv
import os
import sys
import logging

# Add src/ to path so imports work regardless of where the script is run from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from constants import REPO_ROOT
from data import get_dataset
from agent.two_phase_agent import TwoPhaseAgent
from reward_func import db_size_threshold as db_size_threshold_reward
from eval.metrics import exact_match_score
from eval_multihop import split_into_parts


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [run_inference_csv] %(message)s",
)
logger = logging.getLogger("run_inference_csv")

TRAINING_SAMPLING_PARAMS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "vllm_top_k": 4,
    "repetition_penalty": 1.0,
    "max_completion_length": 1024,
    "max_model_len": 8192,
}

CSV_COLUMNS = [
    "step",
    "answer",
    "phase1_prompt",
    "phase2_prompt",
    "phase1_completion_0",
    "phase1_context_0",
    "generated_db_0",
    "phase1_advantage_0",
    "db_size_threshold_0",
    "phase2_completion_0_0",
    "phase2_advantage_0_0",
    "em_accuracy_0_0",
]


def _gold_str(gold) -> str:
    """Match evaluate.py _safe_join_gold: join list answers with newline."""
    if gold is None:
        return ""
    if isinstance(gold, (list, tuple)):
        return "\n".join(str(x) for x in gold)
    return str(gold)


def build_rows(
    global_step: int,
    questions: list[str],
    gold_answers: list,
    contexts_list: list[list[str]],
    answers: list[str],
    traces: list,
    phase1_infos: list[dict],   # indexed by position within this batch
    phase1_dbs: list,           # indexed by position within this batch
) -> list[dict]:
    """Build one CSV row per example."""
    rows = []
    for idx, (question, gold, contexts, answer, trace) in enumerate(
        zip(questions, gold_answers, contexts_list, answers, traces)
    ):
        # ── Phase 1 fields ────────────────────────────────────────────────────
        phase1_info = phase1_infos[idx] if idx < len(phase1_infos) else {}
        phase1_completion = phase1_info.get("raw_text", "")
        # contexts may be list[str] (golden) or list[list[str]] (all/split)
        if contexts and isinstance(contexts[0], list):
            flat_contexts = [p for part in contexts for p in part]
        else:
            flat_contexts = contexts
        phase1_context = "\n\n".join(flat_contexts)

        # generated_db: per-example DB triplets (same format as training logs)
        if idx < len(phase1_dbs):
            generated_db = str(phase1_dbs[idx].database["triplets"])
        else:
            generated_db = str(phase1_info.get("triplets", []))

        # phase1_prompt: reconstruct from template (not stored in _phase1_info)
        # (agent reference not passed in; caller must pass pre-built prompt or
        #  we rely on the fact that it's deterministic from context+question)
        phase1_prompt = phase1_info.get("_phase1_prompt", "")

        db_thresh_vals = db_size_threshold_reward([phase1_completion], contexts=[contexts])
        db_thresh = db_thresh_vals[0] if db_thresh_vals else None

        # ── Phase 2 fields ────────────────────────────────────────────────────
        phase2_initial_prompt = f"Question:\n{question}\nAnswer:\n"

        if trace and trace[0].prompt:
            full_text = trace[0].prompt
            phase2_completion = (
                full_text[len(phase2_initial_prompt):]
                if full_text.startswith(phase2_initial_prompt)
                else full_text
            )
        else:
            phase2_completion = ""

        gold_text = _gold_str(gold)
        em_val = 1 if exact_match_score(answer, gold_text) else 0

        rows.append({
            "step": global_step,
            "answer": gold_text,
            "phase1_prompt": phase1_prompt,
            "phase2_prompt": phase2_initial_prompt,
            "phase1_completion_0": phase1_completion,
            "phase1_context_0": phase1_context,
            "generated_db_0": generated_db,
            "phase1_advantage_0": float("nan"),
            "db_size_threshold_0": db_thresh,
            "phase2_completion_0_0": phase2_completion,
            "phase2_advantage_0_0": float("nan"),
            "em_accuracy_0_0": em_val,
        })
    return rows


def get_contexts(ex, use_contexts: str):
    """Return contexts for one example.

    golden → list[str]  (2 supporting paragraphs)
    all    → list[list[str]]  (all paragraphs split into 5 parts, same as eval_multihop)
    """
    if use_contexts == "golden":
        return ex["golden_contexts"]
    # "all": split all distractor contexts into 5 parts (mirrors eval_multihop.py)
    return split_into_parts(ex.get("contexts") or [], num_parts=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", default="hotpotqa", choices=["hotpotqa", "musique", "2wiki"])
    parser.add_argument("--setting", default="distractor", choices=["distractor", "fullwiki"])
    parser.add_argument("--split", default="validation")
    parser.add_argument("--use-contexts", default="golden", choices=["golden", "all"])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--total-count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--use-train-params", action="store_true")
    parser.add_argument("--concat-all-db", action="store_true",
                        help="Build one unified DB from all examples (recommended eval mode)")
    parser.add_argument("--phase1-prompt-type", default="sft", choices=["sft", "with_question"])
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--similarity-threshold", type=float, default=0.6)
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()

    sampling = dict(TRAINING_SAMPLING_PARAMS) if args.use_train_params else {
        "temperature": 0.0,
        "top_p": 1.0,
        "vllm_top_k": -1,
        "repetition_penalty": 1.0,
        "max_completion_length": 1024,
        "max_model_len": args.max_model_len,
    }
    logger.info("Sampling: %s", sampling)

    # ── Output path ───────────────────────────────────────────────────────────
    if args.output is None:
        if "checkpoint" in args.model_path:
            model_name = (
                args.model_path.split("/")[-2]
                + "-ckpt"
                + args.model_path.split("/")[-1].split("checkpoint-")[-1]
            )
        else:
            model_name = args.model_path.split("/")[-1]
        suffix = "_unified" if args.concat_all_db else ""
        out_dir = os.path.join(REPO_ROOT, "reward_hacking_evaluate")
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(
            out_dir,
            f"{model_name}_{args.dataset}_{args.split}_n{args.total_count}{suffix}.csv",
        )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    logger.info("Output: %s", args.output)

    # ── Dataset ───────────────────────────────────────────────────────────────
    full_dataset = get_dataset(name=args.dataset, setting=args.setting, split=args.split, seed=args.seed)
    end_index = min(args.start_index + args.total_count, len(full_dataset))
    dataset = full_dataset.select(range(args.start_index, end_index))
    logger.info("Processing %d examples (indices %d–%d)", len(dataset), args.start_index, end_index - 1)

    # Collect all questions/contexts/gold answers up front (needed for unified DB build)
    all_questions, all_gold_answers, all_contexts_list = [], [], []
    for ex in dataset:
        all_questions.append(ex["question"])
        all_gold_answers.append(ex["answers"])
        all_contexts_list.append(get_contexts(ex, args.use_contexts))

    # ── Agent ─────────────────────────────────────────────────────────────────
    logger.info("Loading TwoPhaseAgent from %s", args.model_path)
    agent = TwoPhaseAgent(
        model_path=args.model_path,
        phase1_prompt_type=args.phase1_prompt_type,
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        concat_all_db=args.concat_all_db,
        contexts_are_split=(args.use_contexts == "all"),
        **sampling,
    )

    # ── Unified DB mode: build DB once over entire dataset ────────────────────
    # After this call, agent._phase1_info and agent._phase1_dbs are populated
    # for ALL examples (indexed 0..N-1), and agent._unified_db is ready.
    # Subsequent agent.run() calls skip phase 1 and use the unified DB directly.
    if args.concat_all_db:
        logger.info("Building unified DB from %d examples...", len(all_questions))
        agent.build_unified_db_from_dataset(all_questions, all_contexts_list)
        logger.info("Unified DB ready.")

        # Stash phase1 results now — run() won't touch them again
        global_phase1_infos = list(agent._phase1_info)
        global_phase1_dbs   = list(agent._phase1_dbs)

        # Inject phase1_prompt into each info dict for later use in build_rows
        for i, (question, contexts) in enumerate(zip(all_questions, all_contexts_list)):
            if contexts and isinstance(contexts[0], list):
                flat = [p for part in contexts for p in part]
            else:
                flat = contexts
            if agent._phase1_prompt_type == "with_question":
                p1_prompt = agent._phase1_prompt_template.format(
                    context="\n\n".join(flat), question=question
                )
            else:
                p1_prompt = agent._phase1_prompt_template.format(
                    context="\n\n".join(flat)
                )
            global_phase1_infos[i]["_phase1_prompt"] = p1_prompt

    # ── Inference + CSV write ─────────────────────────────────────────────────
    write_header = not os.path.exists(args.output)
    total_em, total_examples = 0, 0

    with open(args.output, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writeheader()

        num_batches = (len(dataset) + args.batch_size - 1) // args.batch_size
        for batch_idx in range(num_batches):
            batch_start = batch_idx * args.batch_size
            batch_end   = min(batch_start + args.batch_size, len(dataset))

            questions    = all_questions[batch_start:batch_end]
            gold_answers = all_gold_answers[batch_start:batch_end]
            contexts_list = all_contexts_list[batch_start:batch_end]

            logger.info("Batch %d/%d: examples %d–%d", batch_idx + 1, num_batches,
                        args.start_index + batch_start, args.start_index + batch_end - 1)

            answers, traces = agent.run(questions, contexts=contexts_list)

            # Resolve phase1 info for this batch
            if args.concat_all_db:
                # Pre-built; slice by batch position in the global list
                batch_phase1_infos = global_phase1_infos[batch_start:batch_end]
                batch_phase1_dbs   = global_phase1_dbs[batch_start:batch_end]
            else:
                # Built fresh by run() → _phase1_build_dbs
                batch_phase1_infos = list(agent._phase1_info)
                batch_phase1_dbs   = list(agent._phase1_dbs)
                # Inject phase1_prompt (not stored in _phase1_info by default)
                for i, (question, contexts) in enumerate(zip(questions, contexts_list)):
                    if contexts and isinstance(contexts[0], list):
                        flat = [p for part in contexts for p in part]
                    else:
                        flat = contexts
                    if agent._phase1_prompt_type == "with_question":
                        p1_prompt = agent._phase1_prompt_template.format(
                            context="\n\n".join(flat), question=question
                        )
                    else:
                        p1_prompt = agent._phase1_prompt_template.format(
                            context="\n\n".join(flat)
                        )
                    if i < len(batch_phase1_infos):
                        batch_phase1_infos[i]["_phase1_prompt"] = p1_prompt

            rows = build_rows(
                global_step=args.start_index + batch_start,
                questions=questions,
                gold_answers=gold_answers,
                contexts_list=contexts_list,
                answers=answers,
                traces=traces,
                phase1_infos=batch_phase1_infos,
                phase1_dbs=batch_phase1_dbs,
            )
            writer.writerows(rows)
            csvfile.flush()

            batch_em = [r["em_accuracy_0_0"] for r in rows if r["em_accuracy_0_0"] is not None]
            total_em += sum(batch_em)
            total_examples += len(rows)
            logger.info(
                "Batch %d/%d — batch EM: %.3f (%d/%d)  running EM: %.3f (%d/%d)",
                batch_idx + 1, num_batches,
                sum(batch_em) / len(batch_em) if batch_em else 0.0, int(sum(batch_em)), len(rows),
                total_em / total_examples, int(total_em), total_examples,
            )

    logger.info("=" * 60)
    logger.info("FINAL EM: %.4f  (%d / %d correct)", total_em / total_examples if total_examples else 0.0,
                int(total_em), total_examples)
    logger.info("CSV: %s", args.output)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
