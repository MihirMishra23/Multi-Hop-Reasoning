#!/usr/bin/env python
"""Debug 8B vs 4B truncation/verbosity in GRPO-style sampling.

Tests two things on the SAME hotpotqa questions, with the SAME sampling config
that GRPO eval uses (temp=1, top_p=0.95, top_k=4, max_tokens=1024, n=32):

  1. PHASE1 (DB generation): given context, generate triplets.
     Reports: % truncated, output token length distribution.

  2. PHASE2 (bare QA): given bare question (no DB, no retrieval),
     generate 32 rollouts. Reports: % truncated, % with <answer> tag,
     mean F1 of valid answers, output length distribution.

Run on 8B-SFT and 4B-SFT separately; the slurm wrappers iterate over both.

Why bare QA (no retrieval)?
  GRPO eval with phase1_prompt_type=sft uses bare QA in phase2 too. We're
  testing whether 8B's verbosity alone (independent of retrieval) causes
  more truncation than 4B.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from datasets import load_dataset
from vllm import LLM, SamplingParams

from eval.metrics import f1_score


PHASE1_PROMPT_TEMPLATE = (
    "Please extract knowledge triplets from the context.\n"
    "Context:\n\n{context}\n\nTriplets:\n"
)


def extract_answer(text: str) -> str:
    for open_tag, close_tag in [("<|answer|>", "<|/answer|>"), ("<answer>", "</answer>")]:
        if open_tag in text:
            try:
                return text.split(open_tag, 1)[1].split(close_tag, 1)[0].strip()
            except IndexError:
                return ""
    return ""


def percentile(xs, p):
    s = sorted(xs)
    idx = int(round((len(s) - 1) * p))
    return s[idx]


def stat_summary(lens, label):
    return (
        f"{label}: n={len(lens)} | "
        f"mean={sum(lens)/len(lens):.1f} | "
        f"p50={percentile(lens, 0.5)} | "
        f"p90={percentile(lens, 0.9)} | "
        f"p99={percentile(lens, 0.99)} | "
        f"max={max(lens)}"
    )


def run_phase1(llm, tokenizer, questions, args):
    """Generate triplets given context. Mirrors GRPO phase1 with K rollouts."""
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=args.k_rollouts,
    )
    prompts = []
    for ex in questions:
        ctx = "\n\n".join(ex["context_text"])
        user_msg = PHASE1_PROMPT_TEMPLATE.format(context=ctx)
        msgs = [{"role": "user", "content": user_msg}]
        prompts.append(
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )

    print(f"[phase1] Generating {args.k_rollouts} rollouts × {len(questions)} questions")
    outputs = llm.generate(prompts, sampling)

    n_total = 0
    n_truncated = 0
    out_lens = []
    samples = []
    for q_idx, output in enumerate(outputs):
        for r in output.outputs:
            n_total += 1
            out_lens.append(len(r.token_ids))
            if r.finish_reason == "length":
                n_truncated += 1
        if q_idx < 3:
            samples.append({
                "question": questions[q_idx]["question"],
                "rollout_0": output.outputs[0].text[:1500],
                "rollout_0_tokens": len(output.outputs[0].token_ids),
                "rollout_0_finish": output.outputs[0].finish_reason,
            })

    print()
    print(f"[phase1] truncated: {n_truncated}/{n_total} = {100*n_truncated/n_total:.1f}%")
    print(f"[phase1] {stat_summary(out_lens, 'token_lens')}")

    return {
        "n_total": n_total,
        "n_truncated": n_truncated,
        "trunc_pct": 100 * n_truncated / n_total,
        "token_lens_mean": sum(out_lens) / len(out_lens),
        "token_lens_p50": percentile(out_lens, 0.5),
        "token_lens_p90": percentile(out_lens, 0.9),
        "token_lens_p99": percentile(out_lens, 0.99),
        "token_lens_max": max(out_lens),
        "samples": samples,
    }


def run_phase2(llm, tokenizer, questions, args):
    """Bare QA: question only, generate N rollouts, check answer tag + F1."""
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=args.n_rollouts,
    )
    prompts = []
    for ex in questions:
        if args.prompt_mode == "raw":
            # Matches standalone eval (two_phase_agent.py:365): bare completion-style.
            p = f"Question:\n{ex['question']}\nAnswer:\n"
        elif args.prompt_mode == "chat_no_think":
            msgs = [{"role": "user", "content": ex["question"]}]
            p = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        else:  # "chat" (current default — same as GRPO trainer)
            msgs = [{"role": "user", "content": ex["question"]}]
            p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(p)

    print(f"[phase2] prompt_mode={args.prompt_mode}")
    print(f"[phase2] Generating {args.n_rollouts} rollouts × {len(questions)} questions")
    print(f"[phase2] sample prompt (first 200 chars):\n{prompts[0][:200]!r}")
    outputs = llm.generate(prompts, sampling)

    n_total = 0
    n_truncated = 0
    n_with_answer = 0
    f1_valid = []
    out_lens = []
    samples = []
    for q_idx, output in enumerate(outputs):
        gold = questions[q_idx]["answer"]
        per_q_with_ans = 0
        for r in output.outputs:
            n_total += 1
            out_lens.append(len(r.token_ids))
            if r.finish_reason == "length":
                n_truncated += 1
            extracted = extract_answer(r.text)
            if extracted:
                n_with_answer += 1
                per_q_with_ans += 1
                f1, _, _ = f1_score(extracted, gold)
                f1_valid.append(f1)
        if q_idx < 3:
            samples.append({
                "question": questions[q_idx]["question"],
                "gold": gold,
                "n_rollouts_with_answer": per_q_with_ans,
                "first_3_rollouts": [
                    {
                        "text": r.text[:800],
                        "n_tokens": len(r.token_ids),
                        "finish_reason": r.finish_reason,
                        "extracted": extract_answer(r.text),
                    }
                    for r in output.outputs[:3]
                ],
            })

    print()
    print(f"[phase2] truncated: {n_truncated}/{n_total} = {100*n_truncated/n_total:.1f}%")
    print(f"[phase2] with <answer> tag: {n_with_answer}/{n_total} = {100*n_with_answer/n_total:.1f}%")
    if f1_valid:
        print(f"[phase2] mean F1 (over rollouts with valid answer): {sum(f1_valid)/len(f1_valid):.4f}")
    print(f"[phase2] {stat_summary(out_lens, 'token_lens')}")

    return {
        "n_total": n_total,
        "n_truncated": n_truncated,
        "trunc_pct": 100 * n_truncated / n_total,
        "n_with_answer": n_with_answer,
        "with_answer_pct": 100 * n_with_answer / n_total,
        "mean_f1_valid": (sum(f1_valid) / len(f1_valid)) if f1_valid else None,
        "token_lens_mean": sum(out_lens) / len(out_lens),
        "token_lens_p50": percentile(out_lens, 0.5),
        "token_lens_p90": percentile(out_lens, 0.9),
        "token_lens_p99": percentile(out_lens, 0.99),
        "token_lens_max": max(out_lens),
        "samples": samples,
    }


def load_questions(n_questions):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    questions = []
    for i in range(n_questions):
        ex = ds[i]
        questions.append({
            "question": ex["question"],
            "answer": ex["answer"],
            "context_text": [
                f"{title}: {' '.join(sents)}"
                for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"])
            ],
        })
    return questions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", default=str(ROOT / "KG_results" / "debug"))
    p.add_argument("--n_questions", type=int, default=20)
    p.add_argument("--n_rollouts", type=int, default=32, help="phase2 rollouts per Q (matches GRPO N=32)")
    p.add_argument("--k_rollouts", type=int, default=4, help="phase1 rollouts per Q (matches GRPO K=4)")
    p.add_argument("--max_tokens", type=int, default=1024, help="matches GRPO max_completion_length")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=4)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85,
                   help="A6000 single-GPU; can be high since no other process")
    p.add_argument("--skip_phase1", action="store_true")
    p.add_argument("--skip_phase2", action="store_true")
    p.add_argument("--prompt_mode", choices=["chat", "raw", "chat_no_think"], default="chat",
                   help="phase2 prompt format: chat=apply_chat_template (GRPO default), "
                        "raw=bare 'Question:...Answer:' (matches standalone eval), "
                        "chat_no_think=chat template with enable_thinking=False")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[config] model_path = {args.model_path}")
    print(f"[config] max_tokens = {args.max_tokens} | "
          f"temperature = {args.temperature} | top_p = {args.top_p} | top_k = {args.top_k}")

    questions = load_questions(args.n_questions)
    print(f"[data] loaded {len(questions)} hotpotqa validation questions")

    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=False,
    )
    tokenizer = llm.get_tokenizer()

    result = {"model_path": args.model_path, "config": vars(args)}
    if not args.skip_phase1:
        print("\n========== PHASE 1: triplet generation ==========")
        result["phase1"] = run_phase1(llm, tokenizer, questions, args)
    if not args.skip_phase2:
        print("\n========== PHASE 2: bare QA ==========")
        result["phase2"] = run_phase2(llm, tokenizer, questions, args)

    model_tag = Path(args.model_path).name
    out_path = Path(args.output_dir) / f"{model_tag}_truncation_debug_{args.prompt_mode}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
