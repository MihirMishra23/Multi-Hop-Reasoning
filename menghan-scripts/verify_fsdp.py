#!/usr/bin/env python
"""Single-GPU replication of GRPO's stochastic 2-phase eval.

Hypothesis to test:
    GRPO step 0 reports `train/rewards/f1_reward/mean = 0.16` for 8B-SFT.
    Standalone (single-GPU, greedy) eval on the same checkpoint gives 0.374.
    Some of the gap is temp=1 sampling noise. The rest may be FSDP-vLLM
    weight-gather corruption during GRPO's colocate inference.

This script runs the SAME stochastic 2-phase pipeline that GRPO runs internally
each step, but on a single GPU with no FSDP. Computes the same f1_reward and
nanmean that GRPO logs as `rewards/f1_reward/mean`.

Reads as:
    Single-GPU result ≈ 0.16 → FSDP is NOT the bug (8B SFT just performs this way
                                under stochastic 2-phase eval).
    Single-GPU result > 0.30 → FSDP-vLLM colocate corrupts weights for 8B.

Usage:
    python menghan-scripts/verify_fsdp.py \\
        --model_path /share/j_sun/lmlm_multihop/checkpoints/main/Qwen3-8B-SFT_hotpotqa_ep3_bsz48_th-1 \\
        --n_questions 20 --n_rollouts 32

Note on K vs N:
    GRPO's command uses K=4 phase1 rollouts × M=8 phase2 rollouts/DB = N=32 total.
    For simplicity this script does N=32 independent phase1+phase2 runs (each
    rollout has its own DB). In expectation this gives the same mean F1.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_dataset

from agent.two_phase_agent import TwoPhaseAgent
from reward_func import f1_reward


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--n_questions", type=int, default=20,
                   help="hotpot validation questions to evaluate (GRPO uses eval_size=100)")
    p.add_argument("--n_rollouts", type=int, default=32,
                   help="rollouts per question (matches GRPO --num_generations_eval)")
    p.add_argument("--max_completion_length", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=4, help="vLLM sampling top_k")
    p.add_argument("--retrieval_threshold", type=float, default=0.6)
    p.add_argument("--retrieval_top_k", type=int, default=4)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--output_dir", default=str(ROOT / "KG_results" / "verify_fsdp"))
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[config] model: {args.model_path}")
    print(f"[config] {args.n_questions} Q × {args.n_rollouts} rollouts = "
          f"{args.n_questions * args.n_rollouts} total")
    print(f"[config] temp={args.temperature} top_p={args.top_p} top_k={args.top_k} "
          f"max_tokens={args.max_completion_length}")
    print(f"[config] retrieval threshold={args.retrieval_threshold} top_k={args.retrieval_top_k}")

    # Load eval data
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    questions = []
    contexts = []
    golds = []
    for i in range(args.n_questions):
        ex = ds[i]
        questions.append(ex["question"])
        ctxs = [
            f"{title}: {' '.join(sents)}"
            for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"])
        ]
        contexts.append(ctxs)
        golds.append(ex["answer"])
    print(f"[data] loaded {len(questions)} hotpotqa val questions")

    # Expand: each Q duplicated N times so each rollout gets its own phase1+phase2
    queries_x_n = []
    contexts_x_n = []
    golds_x_n = []
    for q, c, g in zip(questions, contexts, golds):
        queries_x_n.extend([q] * args.n_rollouts)
        contexts_x_n.extend([c] * args.n_rollouts)
        golds_x_n.extend([g] * args.n_rollouts)

    # Build agent (single-GPU vLLM, no FSDP)
    agent = TwoPhaseAgent(
        model_path=args.model_path,
        phase1_prompt_type="sft",
        top_k=args.retrieval_top_k,
        similarity_threshold=args.retrieval_threshold,
        temperature=args.temperature,
        top_p=args.top_p,
        vllm_top_k=args.top_k,
        max_completion_length=args.max_completion_length,
        max_model_len=args.max_model_len,
    )

    # Run the full 2-phase pipeline (phase1 builds DBs, phase2 with retrieve loop)
    print(f"[generate] running agent on {len(queries_x_n)} queries...")
    answers, traces = agent.run(
        queries_x_n,
        contexts=contexts_x_n,
        max_tokens=args.max_completion_length,
        temperature=args.temperature,
    )

    # Compute f1_reward EXACTLY like GRPO does
    # f1_reward signature: completions, solution → list of float | None
    # We need to feed the FULL completion text (not just extracted answer)
    # The agent returns extracted answers, but we need raw text for the <thinking>/<answer> check
    # Pull raw text from traces
    raw_completions = [t[-1].response for t in traces] if traces else answers
    # Fallback: if traces don't have raw text, just use answers (less accurate)
    rewards = f1_reward(raw_completions, golds_x_n)

    # Stats matching GRPO's logging
    n_total = len(rewards)
    n_none = sum(1 for r in rewards if r is None)
    valid = [r for r in rewards if r is not None]
    nanmean = (sum(valid) / len(valid)) if valid else None

    n_with_answer = sum(1 for a in answers if a)  # non-empty extracted answer

    print()
    print(f"[result] total rollouts: {n_total}")
    print(f"[result] non-empty <answer> extracted: {n_with_answer} ({100*n_with_answer/n_total:.1f}%)")
    print(f"[result] None (no <answer> AND no <thinking>): {n_none} ({100*n_none/n_total:.1f}%)")
    print(f"[result] valid (counted in nanmean): {len(valid)} ({100*len(valid)/n_total:.1f}%)")
    if nanmean is not None:
        print(f"[result] nanmean (≡ GRPO `rewards/f1_reward/mean`): {nanmean:.4f}")
    print()
    print(">>> Compare with GRPO step 0 reward (~0.16 for 8B, ~0.54 for 4B).")
    print(">>> If single-GPU result is much higher than 0.16, FSDP-vLLM is the culprit.")

    # Save
    tag = Path(args.model_path).name
    out_path = out_dir / f"{tag}_verify_fsdp.json"
    samples = []
    for i in range(min(20, len(answers))):
        samples.append({
            "question": queries_x_n[i],
            "gold": golds_x_n[i],
            "extracted_answer": answers[i],
            "reward": rewards[i],
            "raw_completion_first_800": raw_completions[i][:800] if raw_completions[i] else "",
        })
    with open(out_path, "w") as f:
        json.dump({
            "config": vars(args),
            "n_total": n_total,
            "n_with_answer": n_with_answer,
            "n_none": n_none,
            "n_valid": len(valid),
            "nanmean": nanmean,
            "samples": samples,
        }, f, indent=2)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
