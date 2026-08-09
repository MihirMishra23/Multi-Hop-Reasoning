"""Legacy Hermes/ICL3Hop Search-R1 evaluator for Linxi's custom checkpoints.

This is intentionally separate from ``src/agent/search_r1_agent.py``.  The
released PeterJinGo checkpoints use literal ``<search>``/``<information>``
turns, while the custom ``icl3hop_tr1024`` checkpoints were trained with
Hermes ``<tool_call>``/``<tool_response>`` turns.  Mixing those protocols
produces plausible-looking but invalid rollouts.

Reads merged HF Search-R1 checkpoint, runs it on hotpotqa via the same loader
as eval_multihop.py (src.data.get_dataset), generates multi-turn with the
exact prompt format used at training time, calls the retrieval server on
each <tool_call>, parses <answer>, and scores with token-F1 (KBevo's
rewards/hotpotqa_f1.py) + EM (src/eval/metrics.py:exact_match_score).

Output JSON:
{
  "metadata": {model, split, num_samples, max_turns, temperature, top_p, top_k, retrieval_url},
  "summary":  {token_f1_mean, em_mean, n_with_answer, n_turn_avg, n_search_avg},
  "per_question": [
    {"id", "question", "gold", "pred", "answer_text", "score", "em", "n_turns", "trace"}, ...
  ]
}
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

# Make existing modules importable (src/ and rebuttal-search-r1/rewards/).
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "rewards"))

from data import get_dataset  # noqa: E402
from eval.metrics import exact_match_score, f1_score as eval_f1_score  # noqa: E402
from hotpotqa_f1 import compute_score as searchr1_compute_score  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_search_r1_hermes")

EVALUATOR_LINEAGE = {
    "protocol": "hermes_tool_call_icl3hop",
    "restored_from_commit": "6c68b43",
    "restored_from_path": "rebuttal-search-r1/eval_search_r1.py",
}


# Exact system+user wrapping the model was trained with.
SYS_PROMPT = "You are a helpful and harmless assistant."
TOOL_BLOCK = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Searches the web for relevant information based on the given query.", "parameters": {"type": "object", "properties": {"query_list": {"type": "array", "description": "A list of fully-formed semantic queries. The tool will return search results for each query."}}, "required": ["query_list"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""
# --- Prompt variants ----------------------------------------------------------
# Each training run used a different USER_PREFIX (see
# adapt_parquets_for_verl*.py). At eval time we MUST use the same prefix the
# model was trained with, otherwise the policy sees a distribution shift.
#
#   "default"     — original <think>/</think> tag (special-token in Qwen3).
#                   Used by early runs and the 4B@3e-6 finished run.
#   "thinkingtag" — <thinking>/</thinking> (regular 3-token text, no Qwen3
#                   native-thinking conflict). Used by Run D.
#   "icl3hop"     — <thinking> + 3-hop ICL example. Used by Run C.
USER_PREFIX_DEFAULT = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack some "
    "knowledge, you can call the search tool that is available to you; its results will be "
    "returned in a tool response. You can search as many times as you want. If you find no "
    "further external knowledge needed, directly provide the answer inside <answer> and "
    "</answer>, without detailed illustrations. For example, <answer> Beijing </answer>. "
    "Question: "
)
USER_PREFIX_THINKINGTAG = (
    "Answer the given question. You must conduct reasoning inside <thinking> "
    "and </thinking> first every time you get new information. After reasoning, "
    "if you find you lack some knowledge, you can call the search tool that is "
    "available to you; its results will be returned in a tool response. You can "
    "search as many times as you want. If you find no further external knowledge "
    "needed, directly provide the answer inside <answer> and </answer>, without "
    "detailed illustrations. For example, <answer> Beijing </answer>. Question: "
)
USER_PREFIX_ICL3HOP = (
    "Answer the given question. You must conduct reasoning inside <thinking> "
    "and </thinking> first every time you get new information. Each <thinking> "
    "should be ONE BRIEF SENTENCE stating what you need next — not detailed "
    "analysis. After reasoning, if you lack knowledge, call the search tool. "
    "When you have enough information, provide the answer inside <answer> and "
    "</answer> without detailed illustrations.\n"
    "\n"
    "Here is an example showing the expected format and brevity:\n"
    "\n"
    "Question: What is the population of the capital city of the country where "
    "the inventor of the World Wide Web was born?\n"
    "\n"
    "<thinking>I need to find who invented the World Wide Web.</thinking>\n"
    '<tool_call>{"name": "search", "arguments": {"query_list": ["who invented the World Wide Web"]}}</tool_call>\n'
    '<tool_response>Doc 1 (Title: "Tim Berners-Lee"): Sir Timothy John Berners-Lee, '
    "also known as TimBL, is an English computer scientist best known as the "
    "inventor of the World Wide Web.</tool_response>\n"
    "<thinking>Tim Berners-Lee is English, so the country is the United Kingdom. "
    "I need the capital of the UK.</thinking>\n"
    '<tool_call>{"name": "search", "arguments": {"query_list": ["capital of the United Kingdom"]}}</tool_call>\n'
    '<tool_response>Doc 1 (Title: "London"): London is the capital and largest '
    "city of England and the United Kingdom.</tool_response>\n"
    "<thinking>The capital is London. Now I need the population of London."
    "</thinking>\n"
    '<tool_call>{"name": "search", "arguments": {"query_list": ["population of London"]}}</tool_call>\n'
    '<tool_response>Doc 1 (Title: "London"): London has a population of '
    "approximately 9 million people.</tool_response>\n"
    "<thinking>The population is about 9 million.</thinking>\n"
    "<answer>9 million</answer>\n"
    "\n"
    "Now answer this question: "
)
PROMPT_VARIANTS = {
    "default":     USER_PREFIX_DEFAULT,
    "thinkingtag": USER_PREFIX_THINKINGTAG,
    "icl3hop":     USER_PREFIX_ICL3HOP,
}


def build_initial_prompt(tokenizer, question: str, prompt_variant: str,
                         enable_thinking: bool) -> str:
    """Apply Qwen3 chat template to system+user, ending with the assistant turn
    open. `prompt_variant` selects the USER_PREFIX that matches the training run.
    `enable_thinking` controls Qwen3's native thinking-mode injection — set False
    to mirror training when enable_thinking=False was used."""
    user_prefix = PROMPT_VARIANTS[prompt_variant]
    messages = [
        {"role": "system", "content": SYS_PROMPT + "\n\n" + TOOL_BLOCK},
        {"role": "user", "content": user_prefix + question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def parse_tool_call(text: str) -> list[str] | None:
    """Extract query_list from the last <tool_call>...</tool_call> in text. None if invalid."""
    m = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m[-1])
    except json.JSONDecodeError:
        return None
    args = obj.get("arguments", {})
    ql = args.get("query_list")
    if isinstance(ql, list) and all(isinstance(x, str) for x in ql):
        return ql
    if isinstance(ql, str):
        return [ql]
    return None


def _passages2string(retrieval_result: list[dict]) -> str:
    """Mirror verl/tools/utils/search_r1_like_utils.py:_passages2string exactly so the
    model sees the same tool_response body it was trained on."""
    out = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        out += f"Doc {idx + 1} (Title: {title})\n{text}\n\n"
    return out.strip()


def call_retrieval(url: str, queries: list[str], topk: int, timeout: float,
                   max_tool_response_length: int = 512,
                   tool_response_truncate_side: str = "left") -> str:
    """Hit the verl retrieval server with return_scores=True (training-time config),
    then post-process to the formatted multi-doc string + wrap as {"result": ...} JSON.
    Truncates the final JSON string to match verl's training-time truncation
    (verl/experimental/agent_loop/tool_agent_loop.py:_call_tool). Without this,
    popqa retrievals can be 40K+ chars vs the 512 the model was trained on,
    pushing prompts past max_model_len and silently truncating turn 2."""
    try:
        resp = requests.post(
            url,
            json={"queries": queries, "topk": topk, "return_scores": True},
            timeout=timeout,
        )
        resp.raise_for_status()
        api_response = resp.json()
        raw_results = api_response.get("result", [])
        if not raw_results:
            payload = json.dumps({"result": "No search results found."}, ensure_ascii=False)
        else:
            pretty = [_passages2string(r) for r in raw_results]
            payload = json.dumps({"result": "\n---\n".join(pretty)}, ensure_ascii=False)

        # Mirror verl/tool_agent_loop.py truncation exactly.
        if max_tool_response_length and len(payload) > max_tool_response_length:
            if tool_response_truncate_side == "left":
                payload = payload[:max_tool_response_length] + "...(truncated)"
            elif tool_response_truncate_side == "right":
                payload = "(truncated)..." + payload[-max_tool_response_length:]
            else:  # middle
                half = max_tool_response_length // 2
                payload = payload[:half] + "...(truncated)..." + payload[-half:]
        return payload
    except Exception as e:
        return json.dumps({"result": f"Search error: {e}"}, ensure_ascii=False)


def extract_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Late-training drift: model emits answers as a hallucinated
    # <tool_call>{"name":"answer","arguments":{"answer":"..."}}</tool_call>
    # rather than <answer>...</answer>. Pick the LAST such tool_call so
    # multi-search-then-answer traces are scored on the final answer.
    last = ""
    for tc in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL):
        try:
            data = json.loads(tc.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("name") == "answer":
            ans = data.get("arguments", {}).get("answer", "")
            if isinstance(ans, str) and ans.strip():
                last = ans.strip()
    return last


def ordered_id_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(str(value) for value in ids).encode("utf-8")).hexdigest()


def validate_confiqa_manifest(args, qids: list[str]) -> dict[str, Any]:
    """Reject a ConFiQA query/corpus mismatch before reporting a score."""
    if not args.corpus_manifest:
        raise ValueError("ConFiQA Hermes eval requires --corpus_manifest")
    with open(args.corpus_manifest, encoding="utf-8") as stream:
        manifest = json.load(stream)
    selection = manifest.get("dataset_provenance", {}).get("selection", {})
    store_ids = [str(value) for value in selection.get("ordered_ids", [])]
    errors = []
    if selection.get("setting") != args.confiqa_setting:
        errors.append(
            f"setting={selection.get('setting')!r}, expected {args.confiqa_setting!r}"
        )
    if selection.get("seed") != args.seed:
        errors.append(f"seed={selection.get('seed')!r}, expected {args.seed}")
    if selection.get("count") != args.expected_store_samples:
        errors.append(
            f"store count={selection.get('count')!r}, expected {args.expected_store_samples}"
        )
    if len(store_ids) != args.expected_store_samples:
        errors.append(
            f"ordered ID count={len(store_ids)}, expected {args.expected_store_samples}"
        )
    if store_ids[: len(qids)] != [str(value) for value in qids]:
        errors.append("query IDs are not the ordered prefix of the corpus manifest")
    expected_hash = selection.get("ordered_ids_sha256")
    actual_hash = ordered_id_sha256(store_ids)
    if expected_hash != actual_hash:
        errors.append(f"ordered ID hash={expected_hash!r}, recomputed {actual_hash!r}")
    if errors:
        raise ValueError("ConFiQA corpus manifest mismatch: " + "; ".join(errors))
    return manifest


def run_eval(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    log.info("Loading tokenizer + model from %s", args.model_path)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        max_num_seqs=args.max_num_seqs,
        # Disable prefix caching: on B200 + vLLM 0.10.2, prefix caching across
        # multi-turn prompts (system prefix same, body differs by tool_response)
        # appears to cause Turn 2 hangs with longer contexts. Skipping it costs
        # ~10% throughput at start of each turn but stops the hang.
        enable_prefix_caching=False,
    )

    # Dataset — translate (dataset, split_label) to (setting, real_split, start_idx).
    # Mirrors the bash-side mapping in scripts/eval_lmlm_multihop.sh so we eval
    # on the same prompts as the LMLM/two_phase eval.
    DATASET_SPLIT_MAP = {
        ("hotpotqa",  "dev"):           ("distractor", "validation", 0),
        ("hotpotqa",  "train_val1k"):   ("distractor", "train",      82347),
        ("hotpotqa",  "train_train1k"): ("distractor", "train",      89347),
        ("musique",   "dev"):           (None,         "validation", 0),
        ("2wiki",     "dev"):           ("distractor", "validation", 0),
        ("trivia_qa", "dev"):           (None,         "validation", 0),
        ("popqa",     "dev"):           (None,         "test",       0),
        ("confiqa",   "dev"):           (args.confiqa_setting, "test", 0),
        ("confiqa",   "test"):          (args.confiqa_setting, "test", 0),
    }
    key = (args.dataset, args.split)
    if key not in DATASET_SPLIT_MAP:
        raise ValueError(f"Unsupported (dataset, split) = {key}")
    setting, real_split, start_idx = DATASET_SPLIT_MAP[key]
    log.info(
        "Loading %s setting=%s split=%s (label=%s start=%d) seed=%d",
        args.dataset, setting, real_split, args.split, start_idx, args.seed,
    )
    ds = get_dataset(name=args.dataset, setting=setting, split=real_split, seed=args.seed)
    end_idx = start_idx + (args.num_samples if args.num_samples else len(ds) - start_idx)
    end_idx = min(end_idx, len(ds))
    ds = ds.select(range(start_idx, end_idx))
    n = len(ds)
    log.info("Eval set: %d %s questions (indices %d..%d in %s split)",
             n, args.dataset, start_idx, end_idx - 1, real_split)

    questions = [ex["question"] for ex in ds]
    golds = [ex["answers"] for ex in ds]
    qids = [ex["id"] for ex in ds]
    corpus_manifest = None
    if args.dataset == "confiqa":
        corpus_manifest = validate_confiqa_manifest(args, qids)

    # Initial prompts — use the variant + enable_thinking the model was trained with
    log.info("Prompt variant=%s  enable_thinking=%s",
             args.prompt_variant, args.enable_thinking)
    prompts = [
        build_initial_prompt(tok, q, args.prompt_variant, args.enable_thinking)
        for q in questions
    ]
    traces: list[list[dict]] = [[] for _ in range(n)]
    n_turns = [0] * n
    n_search = [0] * n
    finished = [False] * n
    final_text = [""] * n  # the full assistant text (for trace + answer extraction)

    # vLLM stop strings catch the end of a turn. include_stop_str_in_output=True so we can
    # see WHICH stop fired by checking the suffix of the generated text.
    base_stop = ["</tool_call>", "</answer>"]
    sampling = SamplingParams(
        n=1,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else -1,
        max_tokens=args.max_response_length,
        stop=base_stop,
        include_stop_str_in_output=True,
        seed=args.seed,
    )

    for turn in range(args.max_turns):
        active_idx = [i for i in range(n) if not finished[i]]
        if not active_idx:
            break
        log.info("Turn %d / %d — %d active prompts", turn + 1, args.max_turns, len(active_idx))

        # Drop any that have grown past max_model_len budget
        keep = []
        for i in active_idx:
            tok_len = len(tok.encode(prompts[i], add_special_tokens=False))
            if tok_len + args.max_response_length > args.max_model_len:
                finished[i] = True  # truncate trace, scored as no-answer if no <answer>
            else:
                keep.append(i)
        active_idx = keep
        if not active_idx:
            break

        outputs = llm.generate(
            [prompts[i] for i in active_idx], sampling, use_tqdm=(turn == 0)
        )

        for out, i in zip(outputs, active_idx):
            gen = out.outputs[0].text  # generated text only (vLLM strips the prompt by default)
            prompts[i] = prompts[i] + gen
            final_text[i] = final_text[i] + gen
            n_turns[i] = turn + 1

            if gen.rstrip().endswith("</answer>"):
                finished[i] = True
                continue
            if gen.rstrip().endswith("</tool_call>"):
                queries = parse_tool_call(gen)
                trace_entry = {"turn": turn + 1, "kind": "tool_call", "queries": queries}
                if queries is None:
                    trace_entry["error"] = "unparseable_tool_call"
                    traces[i].append(trace_entry)
                    finished[i] = True  # malformed call — give up
                    continue
                retrieval_payload = call_retrieval(
                    args.retrieval_url, queries, args.topk, args.timeout,
                    max_tool_response_length=args.max_tool_response_length,
                    tool_response_truncate_side=args.tool_response_truncate_side,
                )
                trace_entry["n_queries"] = len(queries)
                trace_entry["response_len_chars"] = len(retrieval_payload)
                traces[i].append(trace_entry)
                n_search[i] += 1
                # Append the tool response in the format the model was trained with,
                # then re-open an assistant turn so generation continues. The model's
                # generation ends at "</tool_call>" (vLLM stop string), so we close
                # the assistant turn ourselves with <|im_end|>.
                prompts[i] += (
                    "<|im_end|>\n"
                    "<|im_start|>user\n"
                    f"<tool_response>\n{retrieval_payload}\n</tool_response>"
                    "<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                continue
            # Hit max_tokens without emitting a stop string. Stop here.
            finished[i] = True

    # Score
    per_q = []
    f1_total, em_total, n_answered = 0.0, 0.0, 0
    for i in range(n):
        ans_text = extract_answer(final_text[i])
        # SearchR1 token-F1 (same as training reward; max over multiple gold answers).
        # We pass a synthetic <answer>-wrapped solution so that compute_score sees the
        # extraction our patched extract_answer already did — keeps F1 consistent
        # with EM even when the model used the tool_call answer format.
        sr1_score = searchr1_compute_score(
            data_source="searchR1_hotpotqa",
            solution_str=f"<answer>{ans_text}</answer>" if ans_text else final_text[i],
            ground_truth={"target": golds[i]},
        )
        # Standard EM (any gold matches the prediction after normalization)
        em = 0.0
        for g in golds[i]:
            if exact_match_score(ans_text, g):
                em = 1.0
                break
        if ans_text:
            n_answered += 1
        f1_total += sr1_score
        em_total += em
        row = {
            "id": qids[i],
            "question": questions[i],
            "gold": golds[i],
            "answer_text": ans_text,
            "score_f1": sr1_score,
            "score_em": em,
            "n_turns": n_turns[i],
            "n_search": n_search[i],
            "trace": traces[i],
            # Save the full assistant output for debugging — capped to keep file small
            "raw_output_head": final_text[i][:600],
            "raw_output_tail": final_text[i][-400:] if len(final_text[i]) > 600 else "",
            "raw_output_len": len(final_text[i]),
        }
        if args.save_full_output:
            # Also dump the full assistant text + the full prompt (for debugging
            # what the model saw at each turn). Will inflate the JSON file size
            # substantially — use only with small num_samples.
            row["raw_output_full"] = final_text[i]
            row["prompt_full"] = prompts[i]
        per_q.append(row)

    summary = {
        "n": n,
        "token_f1_mean": f1_total / max(1, n),
        "em_mean": em_total / max(1, n),
        "n_with_answer": n_answered,
        "n_turn_avg": sum(n_turns) / max(1, n),
        "n_search_avg": sum(n_search) / max(1, n),
    }
    log.info("Summary: %s", json.dumps(summary, indent=2))

    out = {
        "metadata": {
            "evaluator": EVALUATOR_LINEAGE,
            "model_path": args.model_path,
            "dataset": args.dataset,
            "split": args.split,
            "num_samples": args.num_samples,
            "max_turns": args.max_turns,
            "max_response_length": args.max_response_length,
            "max_model_len": args.max_model_len,
            "max_tool_response_length": args.max_tool_response_length,
            "tool_response_truncate_side": args.tool_response_truncate_side,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "prompt_variant": args.prompt_variant,
            "enable_thinking": args.enable_thinking,
            "retrieval_url": args.retrieval_url,
            "topk_retrieval": args.topk,
            "seed": args.seed,
            "query_ordered_ids_sha256": ordered_id_sha256(qids),
            "confiqa_setting": args.confiqa_setting if args.dataset == "confiqa" else None,
            "expected_store_samples": (
                args.expected_store_samples if args.dataset == "confiqa" else None
            ),
            "corpus_manifest_path": args.corpus_manifest,
            "corpus_manifest": corpus_manifest,
        },
        "summary": summary,
        "per_question": per_q,
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Wrote %s", args.output_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--dataset", default="hotpotqa",
                   choices=["hotpotqa", "musique", "2wiki", "trivia_qa", "popqa", "confiqa"])
    p.add_argument("--split", default="dev",
                   choices=["dev", "test", "train_val1k", "train_train1k"])
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_path", required=True)
    p.add_argument(
        "--confiqa_setting",
        default="orig",
        choices=[
            "orig", "cf", "cf_100", "cf_500",
            "cf_100_conflict_free", "cf_356_conflict_free",
        ],
    )
    p.add_argument(
        "--corpus_manifest",
        help="Manifest emitted by prepare_confiqa_corpus.py; required for ConFiQA",
    )
    p.add_argument(
        "--expected_store_samples",
        type=int,
        default=1000,
        help="Full retrieval-store size; independent of --num_samples smoke queries",
    )
    # generation — defaults match the Search-R1 a2a training recipe
    # (run_qwen3_1.7b_apples_to_apples.sh): T=1.0, top_p=0.95, top_k=4,
    # max_response_length=2048, max_assistant_turns=5. This also matches
    # the LMLM eval's --use-train-params, so the two pipelines sample
    # under identical distributions for apples-to-apples comparison.
    p.add_argument("--max_turns", type=int, default=5)
    p.add_argument("--max_response_length", type=int, default=2048)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=4)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--enforce_eager", action="store_true",
                   help="Skip vLLM CUDA graph capture (slower, but avoids some hangs)")
    p.add_argument("--max_num_seqs", type=int, default=5,
                   help="Cap on concurrent sequences inside the vLLM engine. "
                        "5 is the known-good ceiling on B200 + vLLM 0.10.2.")
    # retrieval
    p.add_argument("--retrieval_url", default="http://127.0.0.1:8000/retrieve")
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--timeout", type=float, default=30.0)
    # tool_response truncation — MUST match training config to avoid OOD prompts
    p.add_argument("--max_tool_response_length", type=int, default=1024,
                   help="Truncate tool_response to this many chars (matches training). "
                        "Set 0 to disable (NOT recommended — popqa retrievals can be 40K+ chars).")
    p.add_argument("--tool_response_truncate_side", default="left",
                   choices=["left", "right", "middle"],
                   help="Where to keep when truncating; matches verl's "
                        "multi_turn.tool_response_truncate_side training setting.")
    # prompt variant — must match what the model was trained with
    p.add_argument("--prompt_variant", default="default",
                   choices=list(PROMPT_VARIANTS.keys()),
                   help="default: <think> tag (early runs / 4B@3e-6). "
                        "thinkingtag: <thinking> tag, no ICL (Run D). "
                        "icl3hop: <thinking> tag + 3-hop ICL example (Run C).")
    p.add_argument("--save_full_output", action="store_true",
                   help="Dump full per-question trajectory (raw_output_full + prompt_full) "
                        "in addition to head/tail. Use only with small num_samples.")
    p.add_argument("--enable_thinking", type=lambda v: v.lower() == "true",
                   default=False,
                   help="Whether to set enable_thinking=True in the Qwen3 chat "
                        "template. Default False to match training of Runs C/D. "
                        "Set True for default-variant models that trained with "
                        "native thinking enabled.")
    args = p.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
