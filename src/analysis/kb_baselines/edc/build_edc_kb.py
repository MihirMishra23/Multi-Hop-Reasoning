"""Build a static EDC-style KB from dataset contexts using an *untrained* extractor LLM.

This is the rebuttal W2 baseline: same per-passage open IE pipeline as KBEVO Phase-1,
but the extractor is vanilla Qwen3-1.7B-Instruct (no SFT/GRPO), with a context-only
zero-RL prompt + /no_think to suppress Qwen3 thinking output.

The resulting KB JSON has the same schema as the Gemini DBs at
``/share/j_sun/lmlm_multihop/database/gemini/*.json`` (keys: entities,
relationships, return_values, triplets) so it can be loaded directly by
``DatabaseManager.load_database`` and consumed by ``eval_multihop.py
--edc-kb-path ...``.

Apples-to-apples params (locked to KBEVO main eval):
    - 1000 dev examples per dataset, seed=42
    - all 10 contexts split into 5 parts/question
    - vLLM sampling T=1.0, top_p=0.95, top_k=4, max_tokens=1024
    - parse_triplets reused from src/trainer/lmlm_basetrainer.py

Usage:
    PYTHONPATH=src python kb_baselines/edc/build_edc_kb.py \
        --dataset hotpotqa --split dev --setting distractor \
        --n 1000 --seed 42 \
        --extractor Qwen/Qwen3-1.7B \
        --output kb_baselines/edc/output/edc_qwen3_1.7b_hotpotqa_dev_n1000.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Make src/ importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import get_dataset
from trainer.lmlm_basetrainer import parse_triplets


def split_into_parts(items, num_parts):
    """Mirror of src/eval_multihop.py:split_into_parts so chunking is identical."""
    if num_parts <= 0:
        raise ValueError("num_parts must be positive")
    if num_parts >= len(items):
        return [[item] for item in items] + [[] for _ in range(num_parts - len(items))]
    base = len(items) // num_parts
    rem = len(items) % num_parts
    parts, start = [], 0
    for i in range(num_parts):
        size = base + (1 if i < rem else 0)
        parts.append(items[start:start + size])
        start += size
    return parts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["hotpotqa", "musique", "2wiki"])
    p.add_argument("--split", default="dev")
    p.add_argument("--setting", default="distractor")
    p.add_argument("--n", type=int, default=1000, help="examples_to_process")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-parts", type=int, default=5,
                   help="num_parts for context splitting (matches eval_multihop.py)")
    p.add_argument("--use-contexts", default="all", choices=["all", "golden"])
    p.add_argument("--extractor", default="Qwen/Qwen3-1.7B",
                   help="HF id of the extractor model (vanilla Qwen3, NOT an SFT checkpoint)")
    p.add_argument("--prompt-key", default="edc_extract",
                   help="Key in data/prompts/database_creation.json")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--output", required=True)
    p.add_argument("--debug-raw-output", default=None,
                   help="Optional path to dump raw extractor outputs for inspection")
    args = p.parse_args()

    # ── Load prompt template ──
    prompts_path = REPO_ROOT / "data" / "prompts" / "database_creation.json"
    with open(prompts_path) as f:
        prompts = json.load(f)
    if args.prompt_key not in prompts:
        raise KeyError(
            f"prompt_key '{args.prompt_key}' not in {prompts_path}. "
            f"Available: {list(prompts.keys())}"
        )
    template = prompts[args.prompt_key]["prompt"]
    print(f"[EDC] Using prompt '{args.prompt_key}' from {prompts_path}")

    # ── Load dataset (same seed/split as eval) ──
    ds = get_dataset(name=args.dataset, setting=args.setting, split=args.split, seed=args.seed)
    end = min(args.start_index + args.n, len(ds))
    ds = ds.select(range(args.start_index, end))
    print(f"[EDC] {args.dataset}/{args.split}: {len(ds)} examples (seed={args.seed})")

    # ── Build per-part prompts (5 prompts/question by default) ──
    all_prompts = []
    parts_per_q = []  # number of parts per question (so we can group outputs)
    for ex in ds:
        if args.use_contexts == "golden":
            ctx_parts = [ex.get("golden_contexts") or []]
        else:
            contexts = ex.get("contexts") or []
            ctx_parts = split_into_parts(contexts, num_parts=args.num_parts)
        parts_per_q.append(len(ctx_parts))
        for part in ctx_parts:
            ctx_text = "\n\n".join(part)
            if "{question}" in template:
                all_prompts.append(template.format(context=ctx_text, question=ex["question"]))
            else:
                all_prompts.append(template.format(context=ctx_text))

    print(f"[EDC] Generated {len(all_prompts)} prompts ({sum(parts_per_q)/len(parts_per_q):.1f} avg parts/q)")
    print(f"[EDC] First prompt preview:\n{'='*60}\n{all_prompts[0][:1000]}\n{'='*60}")

    # ── Run vLLM extractor ──
    from vllm import LLM, SamplingParams
    print(f"[EDC] Loading extractor: {args.extractor}")
    llm = LLM(
        model=args.extractor,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=42,
        max_model_len=args.max_model_len,
    )
    params = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    print(f"[EDC] Generating with T={args.temperature} top_p={args.top_p} top_k={args.top_k}")
    outputs = llm.generate(all_prompts, params)
    raw_texts = [o.outputs[0].text if o.outputs else "" for o in outputs]
    print(f"[EDC] Generation done. First output preview:\n{'='*60}\n{raw_texts[0][:500]}\n{'='*60}")

    if args.debug_raw_output:
        with open(args.debug_raw_output, "w") as f:
            json.dump({"prompts": all_prompts[:5], "raw": raw_texts[:5]}, f, indent=2)
        print(f"[EDC] Dumped raw sample to {args.debug_raw_output}")

    # ── Parse triplets per output ──
    per_part_triplets = [parse_triplets(t) for t in raw_texts]
    total_raw = sum(len(t) for t in per_part_triplets)
    print(f"[EDC] Parsed {total_raw} raw triplets total ({total_raw/len(per_part_triplets):.1f} per part)")

    # ── Aggregate into unified KB ──
    triplets_set = set()
    for tlist in per_part_triplets:
        for t in tlist:
            triplets_set.add(tuple(t))
    entities = set()
    relationships = set()
    return_values = set()
    for e, r, v in triplets_set:
        entities.add(e)
        relationships.add(r)
        return_values.add(v)

    kb = {
        "entities": sorted(entities),
        "relationships": sorted(relationships),
        "return_values": sorted(return_values),
        "triplets": [list(t) for t in sorted(triplets_set)],
        "_meta": {
            "extractor": args.extractor,
            "prompt_key": args.prompt_key,
            "dataset": args.dataset,
            "split": args.split,
            "setting": args.setting,
            "n": args.n,
            "start_index": args.start_index,
            "seed": args.seed,
            "use_contexts": args.use_contexts,
            "num_parts": args.num_parts,
            "sampling": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_tokens": args.max_tokens,
            },
            "total_prompts": len(all_prompts),
            "total_raw_triplets": total_raw,
            "unique_triplets": len(triplets_set),
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)
    print(f"[EDC] Saved KB to {args.output}")
    print(f"[EDC]   |E|={len(entities)}  |R|={len(relationships)}  "
          f"|V|={len(return_values)}  triplets={len(triplets_set)}")


if __name__ == "__main__":
    main()
