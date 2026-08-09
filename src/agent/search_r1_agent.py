"""Search-R1 evaluation agent backed by vLLM and an HTTP retriever.

The agent only owns Search-R1's multi-turn generation protocol. Dataset loading,
batching, persistence, and scoring remain in ``eval_multihop.py``.

Behaviour is a byte-for-byte reproduction of the release's infer.py — the
prompt, ``_passages2string`` document rendering, ``StopOnSequence`` stop set,
and ``curr_search_template`` prefix newlines all match. Rollout uses vLLM
instead of ``AutoModel.generate`` so a full dev sweep finishes in hours rather
than days; there is no behavioural switch to opt out.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from agent.agent_class import Agent, AgentStep


# Verbatim prompt from the Search-R1 release (evaluate_searchr1.py), which is
# also the format the released checkpoints were RL-trained under. The literal
# <search>/</search> and <information> markers are the tool-calling protocol
# the policy learned.
SEARCHR1_OFFICIAL_PROMPT = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
    "and it will return the top searched results between <information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and "
    "</answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: "
)


def extract_answer(text: str) -> str:
    """Return the first complete answer emitted by the assistant."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_search_query(text: str) -> str | None:
    """Return the last <search> query </search> emitted, as Search-R1 does.

    Mirrors get_query() in the upstream evaluate_searchr1.py: the last match
    wins, because generation is stopped at </search> and the tail is the query
    the model is currently asking for.
    """
    matches = re.findall(r"<search>(.*?)</search>", text, re.DOTALL)
    if not matches:
        return None
    query = matches[-1].strip()
    return query or None


def _passages_to_string(passages: list[dict[str, Any]]) -> str:
    """Render retrieved passages exactly as infer.py::_passages2string does:

        title = content.split("\\n")[0]
        text  = "\\n".join(content.split("\\n")[1:])
        format_reference += f"Doc {idx+1}(Title: {title}) {text}\\n"

    Note "Doc N(Title:" has no space before the paren, the body follows the
    paren after a single space, and each document ends with one newline.
    """
    parts = []
    for index, item in enumerate(passages):
        content = str(item["document"]["contents"])
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        parts.append(f"Doc {index + 1}(Title: {title}) {text}\n")
    return "".join(parts)


def _render_passages(response: dict[str, Any]) -> str:
    results = response.get("result", [])
    if not results:
        return "No search results found."
    # Upstream issues one query per turn and concatenates its documents with
    # no separator between result groups.
    return "".join(_passages_to_string(p) for p in results)


def format_search_response(response: dict[str, Any]) -> str:
    """Render retrieval results as upstream Search-R1's continuation.

    Upstream's ``curr_search_template`` is
        '\\n\\n{output_text}<information>{search_results}</information>\\n\\n'.
    The generated text is appended to the prompt by the caller, so only the
    <information> block and its trailing blank line remain here.
    """
    return f"<information>{_render_passages(response)}</information>\n\n"


_INFORMATION_OPEN = "<information>"
_INFORMATION_CLOSE = "</information>"


def truncate_tool_response(text: str, limit: int, side: str) -> str:
    """Shorten a tool response, preserving <information> tags if present.

    Truncating the raw string used to cut the closing </information> off the
    end. The model then saw a block that was never closed and spent its next
    turn emitting "</information>" to balance it instead of answering, which
    cost ~60% of 3B's answers. Only the payload is shortened here; the tags
    always survive.
    """
    def _clip(body: str) -> str:
        if len(body) <= limit:
            return body
        if side == "left":
            return body[:limit] + "...(truncated)"
        if side == "right":
            return "(truncated)..." + body[-limit:]
        half = limit // 2
        return body[:half] + "...(truncated)..." + body[-half:]

    if text.startswith(_INFORMATION_OPEN) and _INFORMATION_CLOSE in text:
        head = _INFORMATION_OPEN
        rest = text[len(head):]
        body, _, tail = rest.partition(_INFORMATION_CLOSE)
        return head + _clip(body) + _INFORMATION_CLOSE + tail
    return _clip(text)


# infer.py's StopOnSequence checks token-sequence equality against these 6
# variants to catch tokenizations that pack trailing whitespace with </search>.
_INFER_PY_STOPS = [
    "</search>", " </search>",
    "</search>\n", " </search>\n",
    "</search>\n\n", " </search>\n\n",
]


def load_corrected_tokenizer(model_path: str):
    """Persist the tokenizer regex fix so vLLM reloads the corrected tokenizer.

    Transformers flags the legacy Qwen2 tokenizer regex shipped with the Qwen3
    merged checkpoint as incorrect. vLLM 0.10 cannot receive tokenizer kwargs,
    so save the fixed fast tokenizer under the job-local temporary directory and
    pass that snapshot to vLLM.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        fix_mistral_regex=True,
    )
    scratch_root = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
    scratch_root.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = Path(
        tempfile.mkdtemp(prefix="searchr1-tokenizer-", dir=str(scratch_root))
    )
    tokenizer.save_pretrained(tokenizer_dir)
    return tokenizer, str(tokenizer_dir)


class SearchR1Agent(Agent):
    """Run batched Search-R1 trajectories while preserving the shared Agent API."""

    def __init__(
        self,
        model_path: str,
        retrieval_url: str,
        max_steps: int = 5,
        retrieval_top_k: int = 3,
        retrieval_timeout: float = 30.0,
        retrieval_workers: int = 32,
        top_p: float = 0.95,
        sampling_top_k: int = -1,
        max_model_len: int = 3072,
        max_tool_response_length: int = 4096,
        tool_response_truncate_side: str = "left",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        # The released Search-R1 checkpoints ship config.json with
        # torch_dtype=float32, because verl's FSDP saves the fp32 master
        # weights. Training and rollout both ran under bf16 autocast, so bf16
        # is the dtype these weights were actually produced under — and it
        # halves the footprint (7B: 29GB -> ~15GB), which is what makes them
        # fit on a 48GB card. Pass dtype=None to follow config.json instead.
        dtype: str | None = "bfloat16",
        max_num_seqs: int | None = None,
        enforce_eager: bool = False,
        seed: int = 0,
        tokenizer: Any | None = None,
        engine: Any | None = None,
        sampling_params_cls: Callable[..., Any] | None = None,
        **_: Any,
    ) -> None:
        if not model_path:
            raise ValueError("--model-path is required for Search-R1 evaluation")

        engine_tokenizer_path = model_path
        if tokenizer is None:
            tokenizer, engine_tokenizer_path = load_corrected_tokenizer(model_path)
        if engine is None or sampling_params_cls is None:
            from vllm import LLM, SamplingParams

            sampling_params_cls = sampling_params_cls or SamplingParams
            if engine is None:
                engine_kwargs = {
                    "model": model_path,
                    "tokenizer": engine_tokenizer_path,
                    "tensor_parallel_size": tensor_parallel_size,
                    "gpu_memory_utilization": gpu_memory_utilization,
                    "max_model_len": max_model_len,
                    "enforce_eager": enforce_eager,
                    "seed": seed,
                }
                if dtype is not None:
                    engine_kwargs["dtype"] = dtype
                if max_num_seqs is not None:
                    engine_kwargs["max_num_seqs"] = max_num_seqs
                engine = LLM(**engine_kwargs)

        self.model_path = model_path
        self.tokenizer_source = model_path
        self.fix_mistral_regex = True
        self.retrieval_url = retrieval_url
        self.max_steps = max_steps
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_timeout = retrieval_timeout
        self.retrieval_workers = retrieval_workers
        self.top_p = top_p
        self.sampling_top_k = sampling_top_k
        self.max_model_len = max_model_len
        self.max_tool_response_length = max_tool_response_length
        self.tool_response_truncate_side = tool_response_truncate_side
        self.tok = tokenizer
        self.llm = engine
        self._sampling_params_cls = sampling_params_cls

    def _initial_prompt(self, question: str) -> str:
        # Upstream normalises the question to end with '?' before templating.
        # A single user turn with no system prompt and no tool schema; the
        # protocol lives entirely in the prompt text.
        question = question.strip()
        if question and question[-1] != "?":
            question += "?"
        return self.tok.apply_chat_template(
            [{"role": "user", "content": SEARCHR1_OFFICIAL_PROMPT + question}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def _token_ids(self, text: str) -> list[int]:
        return list(self.tok.encode(text, add_special_tokens=False))

    def _decode_completion(self, completion: Any) -> tuple[str, list[int]]:
        token_ids = getattr(completion, "token_ids", None)
        if token_ids is None:
            text = completion.text
            return text, self._token_ids(text)
        token_ids = list(token_ids)
        return self.tok.decode(token_ids, skip_special_tokens=False), token_ids

    def _retrieve(self, queries: list[Any]) -> str:
        payload = json.dumps(
            {"queries": queries, "topk": self.retrieval_top_k, "return_scores": True}
        ).encode("utf-8")
        request = urllib.request.Request(
            self.retrieval_url,
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.retrieval_timeout) as response:
                    result = format_search_response(
                        json.loads(response.read().decode("utf-8")),
                    )
                return truncate_tool_response(
                    result,
                    self.max_tool_response_length,
                    self.tool_response_truncate_side,
                )
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(attempt + 1)
        error = f"<information>Search error: {last_error}</information>\n\n"
        return truncate_tool_response(
            error,
            self.max_tool_response_length,
            self.tool_response_truncate_side,
        )

    def _retrieve_many(self, query_lists: list[list[Any]]) -> list[str]:
        if not query_lists:
            return []
        workers = min(self.retrieval_workers, len(query_lists))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(self._retrieve, query_lists))

    def run(
        self,
        queries: list[str] | str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        **_: Any,
    ) -> tuple[list[str], list[list[AgentStep]]]:
        """Generate Search-R1 answers for one evaluator batch.

        ``max_tokens`` is the total trajectory budget, including tool-response
        continuations, matching veRL's multi-turn response-length semantics.
        """
        if isinstance(queries, str):
            queries = [queries]
        response_limit = max_tokens or 2048
        prompts = [self._initial_prompt(question) for question in queries]
        assistant_text = [""] * len(queries)
        response_tokens = [0] * len(queries)
        traces: list[list[AgentStep]] = [[] for _ in queries]
        finished = [False] * len(queries)

        for turn in range(self.max_steps):
            active: list[int] = []
            sampling_params = []
            prompt_snapshots: dict[int, str] = {}
            for index, prompt in enumerate(prompts):
                if finished[index]:
                    continue
                prompt_tokens = len(self._token_ids(prompt))
                budget = min(
                    response_limit - response_tokens[index],
                    self.max_model_len - prompt_tokens,
                )
                if budget <= 0:
                    finished[index] = True
                    continue
                active.append(index)
                prompt_snapshots[index] = prompt
                # Upstream stops generation at </search> so the query can be
                # executed and the <information> block spliced in. Without
                # this the model keeps going and hallucinates its own
                # search results instead of waiting for real ones.
                sampling_params.append(self._sampling_params_cls(
                    n=1,
                    temperature=temperature,
                    top_p=self.top_p,
                    top_k=self.sampling_top_k,
                    max_tokens=budget,
                    stop=_INFER_PY_STOPS,
                    include_stop_str_in_output=True,
                ))

            if not active:
                break

            outputs = self.llm.generate(
                [prompts[index] for index in active],
                sampling_params,
                use_tqdm=False,
            )
            pending: list[tuple[int, list[Any], AgentStep]] = []

            for output, index in zip(outputs, active):
                generated, generated_ids = self._decode_completion(output.outputs[0])
                # infer.py's curr_search_template = '\n\n{output_text}<information>...'
                # inserts '\n\n' before every turn's output when concatenating
                # into the growing prompt (including turn 0). Reproduce that
                # here so the next-turn context matches infer.py byte-for-byte.
                prompts[index] += "\n\n" + generated
                assistant_text[index] += generated
                response_tokens[index] += len(generated_ids)
                search_query = parse_search_query(generated)
                tool_call = (
                    {"name": "search", "arguments": {"query_list": [search_query]}}
                    if search_query
                    else None
                )

                exhausted = (
                    response_tokens[index] >= response_limit
                    or len(self._token_ids(prompts[index])) >= self.max_model_len
                )
                if tool_call is None or turn + 1 >= self.max_steps or exhausted:
                    traces[index].append(
                        AgentStep(
                            prompt=prompt_snapshots[index],
                            answer=generated,
                            action="finish",
                            error=(
                                "trajectory budget exhausted before tool execution"
                                if tool_call is not None and exhausted
                                else "maximum turns reached before tool execution"
                                if tool_call is not None and turn + 1 >= self.max_steps
                                else None
                            ),
                        )
                    )
                    finished[index] = True
                    continue

                arguments = tool_call["arguments"]
                query_list = arguments["query_list"]
                step = AgentStep(
                    prompt=prompt_snapshots[index],
                    answer=generated,
                    action="toolcall",
                    tool_name=tool_call["name"],
                    tool_args=arguments,
                )
                pending.append((index, query_list, step))

            retrieval_inputs = [queries for _, queries, _ in pending if queries]
            retrieval_results = iter(self._retrieve_many(retrieval_inputs))
            for index, tool_queries, step in pending:
                result = next(retrieval_results) if tool_queries else step.tool_result
                step.tool_result = result
                # Official protocol: the <information> block is already wrapped
                # and appended verbatim to the prompt, keeping the trajectory a
                # single assistant turn — what the policy saw during training.
                continuation_ids = self._token_ids(result)
                if (
                    response_tokens[index] + len(continuation_ids) >= response_limit
                    or len(self._token_ids(prompts[index])) + len(continuation_ids)
                    >= self.max_model_len
                ):
                    step.error = step.error or "tool response exceeded trajectory budget"
                    finished[index] = True
                else:
                    prompts[index] += result
                    response_tokens[index] += len(continuation_ids)
                traces[index].append(step)

        answers = [extract_answer(text) for text in assistant_text]
        return answers, traces
