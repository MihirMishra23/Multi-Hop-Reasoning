"""End-to-end rollout test: model → tool call → search → tool response → answer.

Simulates exactly one Search-R1 rollout per question, **without** verl's training
machinery. The point is to answer: for a given question, can the base model
- emit a parseable `<tool_call>` in the format verl/sglang expect,
- have that call dispatched to the real retrieval server through verl's
  `SearchTool`,
- consume the returned docs in a subsequent turn, and
- emit a final `<answer>...</answer>` block.

If this works for at least some questions, the RL loop has something to
bootstrap from. If it fails on every question, RL will collapse to 0 reward
because no rollout ever produces a tool call to learn from.

Run (retrieval server must be up on :8000):
    /scratch/lz586/envs/searchr1-sgl/bin/python \
        rebuttal-search-r1/test_end_to_end_rollout.py

Env overrides:
    MODEL_PATH        default: Qwen/Qwen3-1.7B
    TOOL_CONFIG       default: rebuttal-search-r1/verl_config/tool_config/search_tool_config.yaml
    CUDA_VISIBLE_DEVICES  pick a GPU not in use by the retrieval server
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TOOL_CONFIG = SCRIPT_DIR / "verl_config" / "tool_config" / "search_tool_config.yaml"

# Must mirror adapt_parquets_for_verl.py exactly — same prompt the trainer sees.
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

# A handful of HotpotQA-flavored multi-hop questions where retrieval should help
TEST_QUESTIONS = [
    "Which film director was born first: Akira Kurosawa or Stanley Kubrick?",
    "What mountain range contains the national forest that includes the Wilson Creek area?",
    "What is the population in 2010 of the city where the 2015 New York Giants played their home games?",
]

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
ANSWER_RE    = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _hdr(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)
def _sub(t): print("\n" + "-" * 78 + f"\n{t}\n" + "-" * 78, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Real verl SearchTool — same path the trainer would invoke.
# ─────────────────────────────────────────────────────────────────────────────
class SearchToolBridge:
    """Wraps verl.tools.search_tool.SearchTool so we can call it synchronously
    from the generation loop, reusing one instance_id per question."""

    def __init__(self, tool_config_path: Path):
        import ray
        from verl.tools.utils.tool_registry import initialize_tools_from_config

        if not ray.is_initialized():
            ray.init(local_mode=True, ignore_reinit_error=True, log_to_driver=False)
        self.tool = initialize_tools_from_config(str(tool_config_path))[0]
        self._loop = asyncio.new_event_loop()
        self._instance_id = None

    def new_instance(self):
        async def _c():
            iid, _ = await self.tool.create()
            return iid
        self._instance_id = self._loop.run_until_complete(_c())
        return self._instance_id

    def execute(self, query_list):
        async def _e():
            return await self.tool.execute(self._instance_id, {"query_list": query_list})
        response, reward, metrics = self._loop.run_until_complete(_e())
        return response, metrics

    def close(self):
        if self._instance_id is not None:
            self._loop.run_until_complete(self.tool.release(self._instance_id))
        self._loop.close()


# ─────────────────────────────────────────────────────────────────────────────
# Generation step. We DON'T re-render the chat template after every turn; we
# splice the tool response into the prompt the same way Qwen3's template would,
# so what the model sees on turn N is byte-identical to what verl/sglang would
# show it. Stop sequences cut generation at </tool_call> or </answer>.
# ─────────────────────────────────────────────────────────────────────────────
SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Searches the web for relevant information based on the given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_list": {
                    "type": "array",
                    "items": {"type": "string"},  # NB: 'items' (plural). yaml has 'item' but verl drops it anyway.
                    "description": "A list of fully-formed semantic queries.",
                },
            },
            "required": ["query_list"],
        },
    },
}


def render_initial_prompt(tokenizer, question):
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user",   "content": USER_PREFIX + question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tools=[SEARCH_TOOL_SCHEMA],
        add_generation_prompt=True,
        tokenize=False,
    )


def append_tool_response(prompt, assistant_text, tool_response_text):
    """Splice an assistant turn + tool response back onto the prompt in Qwen3's
    expected format and open a fresh assistant turn. Mirrors what Qwen3's chat
    template emits for a `tool_calls`/`tool` message pair."""
    return (
        prompt
        + assistant_text
        + "<|im_end|>\n"
        + "<|im_start|>user\n"
        + "<tool_response>\n"
        + tool_response_text
        + "\n</tool_response><|im_end|>\n"
        + "<|im_start|>assistant\n"
    )


def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_p, top_k, stop_strings):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
            stop_strings=stop_strings,
            tokenizer=tokenizer,
        )
    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=False)


# ─────────────────────────────────────────────────────────────────────────────
# Single-question rollout
# ─────────────────────────────────────────────────────────────────────────────
def run_one(question, model, tokenizer, bridge, args):
    _sub(f"Q: {question}")
    bridge.new_instance()
    prompt = render_initial_prompt(tokenizer, question)

    transcript = []          # list of (role, text)
    n_tool_calls = 0
    saw_valid_tool_call = False
    saw_valid_search_result = False
    final_answer = None

    for turn in range(args.max_turns):
        gen = generate(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            stop_strings=["</tool_call>", "</answer>", "<|im_end|>"],
        )
        # Strip the trailing <|im_end|> if the tokenizer included it.
        gen_clean = gen.replace("<|im_end|>", "").rstrip()
        transcript.append(("assistant", gen_clean))
        print(f"  [turn {turn+1}] assistant ({len(gen_clean)} chars):")
        print("    " + gen_clean[:600].replace("\n", "\n    "))

        # 1. answer?
        am = ANSWER_RE.search(gen_clean)
        if am:
            final_answer = am.group(1).strip()
            print(f"  [turn {turn+1}] FINAL ANSWER: {final_answer!r}")
            break

        # 2. tool call?
        tm = TOOL_CALL_RE.search(gen_clean)
        if not tm:
            print(f"  [turn {turn+1}] no <tool_call> and no <answer>; giving up.")
            break

        n_tool_calls += 1
        raw = tm.group(1)
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  [turn {turn+1}] tool_call JSON invalid ({e}); raw={raw[:200]!r}")
            break
        if obj.get("name") != "search":
            print(f"  [turn {turn+1}] tool_call name != 'search' ({obj.get('name')!r})")
            break
        args_dict = obj.get("arguments", {})
        ql = args_dict.get("query_list")
        if not isinstance(ql, list) or len(ql) == 0:
            print(f"  [turn {turn+1}] missing/empty query_list: {args_dict!r}")
            break
        saw_valid_tool_call = True
        print(f"  [turn {turn+1}] PARSED tool_call.queries = {ql}")

        # 3. dispatch via real verl SearchTool
        t0 = time.time()
        response, metrics = bridge.execute(ql)
        dt = time.time() - t0
        print(f"  [turn {turn+1}] search ({dt:.2f}s) status={metrics.get('status')} "
              f"total_results={metrics.get('total_results')}")
        if metrics.get("status") != "success":
            print(f"  [turn {turn+1}] search failed; api_error={metrics.get('api_request_error')}")
            break
        saw_valid_search_result = True

        # 4. splice tool response back into prompt and continue
        prompt = append_tool_response(prompt, gen_clean, response.text)
        transcript.append(("tool", response.text))

    return {
        "question": question,
        "n_tool_calls": n_tool_calls,
        "saw_valid_tool_call": saw_valid_tool_call,
        "saw_valid_search_result": saw_valid_search_result,
        "final_answer": final_answer,
        "transcript": transcript,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH", "Qwen/Qwen3-1.7B"))
    ap.add_argument("--tool-config", default=os.environ.get("TOOL_CONFIG", str(DEFAULT_TOOL_CONFIG)))
    ap.add_argument("--max-turns", type=int, default=4)         # matches apples_to_apples
    ap.add_argument("--max-new-tokens", type=int, default=512)
    # Match the apples_to_apples rollout sampling: temp=1.0, top_p=0.95, top_k=4.
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--questions", nargs="*", default=None, help="override TEST_QUESTIONS")
    args = ap.parse_args()

    _hdr("End-to-end rollout test (Qwen3 + verl SearchTool + retrieval server)")
    print(f"model        = {args.model}")
    print(f"tool_config  = {args.tool_config}")
    print(f"max_turns    = {args.max_turns}    max_new_tokens={args.max_new_tokens}")
    print(f"sampling     = temp={args.temperature} top_p={args.top_p} top_k={args.top_k}")

    print("\nLoading tokenizer + model ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  loaded on {model.device}", flush=True)

    # Quick visibility: what does the rendered prompt look like?
    sample_prompt = render_initial_prompt(tok, TEST_QUESTIONS[0])
    _sub("Rendered initial prompt (first 2200 chars)")
    print(sample_prompt[:2200])

    print("\nInitializing verl SearchTool ...", flush=True)
    bridge = SearchToolBridge(Path(args.tool_config))

    questions = args.questions if args.questions else TEST_QUESTIONS
    results = []
    try:
        for q in questions:
            results.append(run_one(q, model, tok, bridge, args))
    finally:
        bridge.close()

    _hdr("SUMMARY")
    n = len(results)
    cnt_valid_call   = sum(r["saw_valid_tool_call"]     for r in results)
    cnt_valid_search = sum(r["saw_valid_search_result"] for r in results)
    cnt_answered     = sum(r["final_answer"] is not None for r in results)
    cnt_full_loop    = sum(r["saw_valid_search_result"] and r["final_answer"] is not None for r in results)
    for r in results:
        flag = "+" if r["saw_valid_search_result"] and r["final_answer"] else " "
        print(f"  {flag} q={r['question'][:60]!r:62}  "
              f"calls={r['n_tool_calls']} valid_call={r['saw_valid_tool_call']} "
              f"got_docs={r['saw_valid_search_result']} answer={r['final_answer']!r}")
    print()
    print(f"  valid tool call    : {cnt_valid_call}/{n}  ({100*cnt_valid_call/n:.0f}%)")
    print(f"  search returned ok : {cnt_valid_search}/{n}  ({100*cnt_valid_search/n:.0f}%)")
    print(f"  final answer       : {cnt_answered}/{n}  ({100*cnt_answered/n:.0f}%)")
    print(f"  full pipeline ✓    : {cnt_full_loop}/{n}  ({100*cnt_full_loop/n:.0f}%)")

    if cnt_valid_call == 0:
        print("\nVERDICT: model cannot emit a parseable <tool_call>. "
              "RL has nothing to bootstrap; fix the prompt/format or use Qwen3-Instruct.")
        sys.exit(2)
    if cnt_full_loop == 0:
        print("\nVERDICT: model emits tool calls and search fires, but no rollout "
              "completes the full loop with an <answer>. RL may still bootstrap "
              "but the reward signal will be very sparse.")
        sys.exit(3)
    print(f"\nVERDICT: {cnt_full_loop}/{n} questions completed the full call→search→answer "
          "loop. RL should bootstrap from these.")


if __name__ == "__main__":
    main()
