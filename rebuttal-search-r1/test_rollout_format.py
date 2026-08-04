"""Sanity check: does the base model's initial rollout emit a parseable tool call?

Loads the same model + chat template + tool definition that verl uses, runs a
single generation on a HotpotQA-style question, and checks whether the model
outputs a `<tool_call>{"name":"search","arguments":{...}}</tool_call>` block
that verl's parser would actually accept.

This isolates the format question from the RL loop: if even greedy decoding
from priors can't produce a valid tool call, RL has nothing to bootstrap from.

Usage:
    python test_rollout_format.py                       # vanilla Qwen3-1.7B base
    python test_rollout_format.py --model Qwen/Qwen3-1.7B-Instruct
    python test_rollout_format.py --enable-thinking     # Qwen3 native <think>
    python test_rollout_format.py --n 4                 # sample N completions
"""

import argparse
import json
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Mirror adapt_parquets_for_verl.py — keep these in sync.
SYSTEM_CONTENT = "You are a helpful and harmless assistant."
USER_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and "
    "</think> first every time you get new information. After reasoning, if you "
    "find you lack some knowledge, you can call the search tool that is available "
    "to you; its results will be returned in a tool response. You can search as "
    "many times as you want. If you find no further external knowledge needed, "
    "directly provide the answer inside <answer> and </answer>, without detailed "
    "illustrations. For example, <answer> Beijing </answer>. Question: "
)

# The OLD (broken) user prefix that was in adapt_parquets_for_verl.py before
# we fixed it — what the verl training run was actually seeing. Describes
# the tool-call format as free-text `<tool_call> query </tool_call>`, which
# is NOT what verl's parser accepts (it wants JSON).
USER_PREFIX_OLD = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <tool_call> query </tool_call> "
    "and it will return the top searched results between <tool_response> and "
    "</tool_response>. You can search as many times as your want. If you find no "
    "further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>. Question: "
)

# Tool schema — must match verl_config/tool_config/search_tool_config.yaml
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Searches the web for relevant information based on the given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_list": {
                    "type": "array",
                    "item": {"type": "string"},
                    "description": "A list of fully-formed semantic queries. The tool will return search results for each query.",
                },
            },
            "required": ["query_list"],
        },
    },
}

# A handful of HotpotQA-flavored multi-hop questions
TEST_QUESTIONS = [
    "When did one of the makers of the film October: Ten Days That Shook the World win the Stalin Prizes?",
    "What mountain range contains the national forest that includes the Wilson Creek area?",
    "What is the population in 2010 of the city where the 2015 New York Giants played their home games?",
    "Which film director was born first: Akira Kurosawa or Stanley Kubrick?",
]

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def render_prompt(tokenizer, question, enable_thinking=None,
                  use_old_prompt=False, inject_tools=True):
    prefix = USER_PREFIX_OLD if use_old_prompt else USER_PREFIX
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": prefix + question},
    ]
    kwargs = dict(add_generation_prompt=True, tokenize=False)
    if inject_tools:
        kwargs["tools"] = [SEARCH_TOOL]
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **kwargs)


def analyze(text):
    findings = {"has_think": False, "has_tool_call": False, "tool_call_json_valid": False,
                "tool_name_correct": False, "has_query_list": False, "has_answer": False,
                "tool_call_raw": None, "tool_call_parsed": None, "answer": None}
    if THINK_RE.search(text):
        findings["has_think"] = True
    m = TOOL_CALL_RE.search(text)
    if m:
        findings["has_tool_call"] = True
        raw = m.group(1)
        findings["tool_call_raw"] = raw
        try:
            obj = json.loads(raw)
            findings["tool_call_json_valid"] = True
            findings["tool_call_parsed"] = obj
            if obj.get("name") == "search":
                findings["tool_name_correct"] = True
            args = obj.get("arguments", {})
            if isinstance(args, dict) and isinstance(args.get("query_list"), list):
                findings["has_query_list"] = True
        except json.JSONDecodeError:
            pass
    a = ANSWER_RE.search(text)
    if a:
        findings["has_answer"] = True
        findings["answer"] = a.group(1).strip()
    return findings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n", type=int, default=2, help="completions per question")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    # Qwen3 thinking-mode recommended: temp=0.6, top_p=0.95, top_k=20
    # Qwen3 non-thinking recommended: temp=0.7, top_p=0.8, top_k=20
    # verl rollout uses: temp=1.0, top_p=0.95, top_k=4 (NOT Qwen3-recommended)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--enable-thinking", action="store_true",
                    help="Qwen3-Instruct only — pass enable_thinking=True to chat template")
    ap.add_argument("--disable-thinking", action="store_true",
                    help="Qwen3-Instruct only — pass enable_thinking=False")
    ap.add_argument("--old-prompt", action="store_true",
                    help="Use the BROKEN user prefix that was in adapt_parquets_for_verl.py before the fix")
    ap.add_argument("--no-tools-in-template", action="store_true",
                    help="Do NOT pass tools=[...] to apply_chat_template (simulates a code path that forgets to inject the tool spec)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    enable_thinking = None
    if args.enable_thinking:
        enable_thinking = True
    if args.disable_thinking:
        enable_thinking = False

    print(f"Loading {args.model} on {args.device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True,
    )
    model.eval()

    print(f"Settings: old_prompt={args.old_prompt} inject_tools={not args.no_tools_in_template} "
          f"temperature={args.temperature} top_p={args.top_p} top_k={args.top_k}",
          flush=True)
    sample = render_prompt(tok, TEST_QUESTIONS[0], enable_thinking=enable_thinking,
                           use_old_prompt=args.old_prompt,
                           inject_tools=not args.no_tools_in_template)
    print("=" * 80)
    print("RENDERED PROMPT (first 2500 chars):")
    print(sample[:2500])
    print("=" * 80)

    totals = {"has_think": 0, "has_tool_call": 0, "tool_call_json_valid": 0,
              "tool_name_correct": 0, "has_query_list": 0, "has_answer": 0}
    n_runs = 0
    for q in TEST_QUESTIONS:
        prompt = render_prompt(tok, q, enable_thinking=enable_thinking,
                               use_old_prompt=args.old_prompt,
                               inject_tools=not args.no_tools_in_template)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        for trial in range(args.n):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    pad_token_id=tok.eos_token_id,
                )
            gen = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False)
            f = analyze(gen)
            n_runs += 1
            for k in totals:
                totals[k] += int(f[k])
            print("-" * 80)
            print(f"Q: {q}")
            print(f"  has_think={f['has_think']} | has_tool_call={f['has_tool_call']} | "
                  f"json_valid={f['tool_call_json_valid']} | name=search={f['tool_name_correct']} | "
                  f"query_list={f['has_query_list']} | has_answer={f['has_answer']}")
            if f["tool_call_raw"]:
                print(f"  tool_call_raw: {f['tool_call_raw'][:300]}")
            if f["answer"]:
                print(f"  answer: {f['answer'][:200]}")
            if not f["has_tool_call"] and not f["has_answer"]:
                print(f"  completion[:400]: {gen[:400]!r}")

    print("=" * 80)
    print(f"SUMMARY across {n_runs} completions:")
    for k, v in totals.items():
        print(f"  {k:>22}: {v}/{n_runs}  ({100*v/n_runs:.0f}%)")
    print("=" * 80)
    # Hard verdict
    if totals["has_query_list"] == 0:
        print("VERDICT: model CANNOT bootstrap — no parseable tool call from priors.")
        sys.exit(1)
    elif totals["has_query_list"] < n_runs * 0.25:
        print("VERDICT: weak bootstrap — fewer than 25% of rollouts produce a valid tool call.")
        sys.exit(2)
    else:
        print("VERDICT: model can bootstrap — at least 25% of rollouts produce a valid tool call.")


if __name__ == "__main__":
    main()
