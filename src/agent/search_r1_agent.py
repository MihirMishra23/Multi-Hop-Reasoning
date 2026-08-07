"""Search-R1 evaluation agent backed by vLLM and an HTTP retriever.

The agent only owns Search-R1's multi-turn generation protocol. Dataset loading,
batching, persistence, and scoring remain in ``eval_multihop.py``.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from agent.agent_class import Agent, AgentStep


SYSTEM_PROMPT = "You are a helpful and harmless assistant."

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
                    "description": (
                        "A list of fully-formed semantic queries. The tool will return "
                        "search results for each query."
                    ),
                }
            },
            "required": ["query_list"],
        },
    },
}

# Verbatim prompt from the Search-R1 release (evaluate_searchr1.py), which is
# also the format the released checkpoints were RL-trained under. The literal
# <search>/</search> and <information> markers are the tool-calling protocol the
# policy learned; paraphrasing them (as the tool_call variants below do) makes
# the model emit something else and the retriever is never invoked.
SEARCHR1_OFFICIAL_PROMPT = (
    "Answer the given question. "
    "You must conduct reasoning inside <think> and </think> first every time you get new information. "
    "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
    "and it will return the top searched results between <information> and </information>. "
    "You can search as many times as your want. "
    "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and "
    "</answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: "
)

# Variants that speak the official <search> protocol. Everything else uses the
# Hermes <tool_call> schema, which only suits models trained for that protocol.
OFFICIAL_PROTOCOL_VARIANTS = {"default"}

PROMPT_VARIANTS = {
    "default": SEARCHR1_OFFICIAL_PROMPT,
    # Paraphrased tool_call variants. NOTE: these do not work with the released
    # PeterJinGo Search-R1 checkpoints — measured 0 retrievals in 20 samples,
    # EM roughly a third of the official number — because those checkpoints only
    # ever saw the <search> protocol. Keep them for models trained on the
    # Hermes tool_call schema.
    "toolcall": (
        "Answer the given question. You must conduct reasoning inside <think> and </think> "
        "first every time you get new information. After reasoning, if you find you lack some "
        "knowledge, you can call the search tool that is available to you; its results will be "
        "returned in a tool response. You can search as many times as you want. If you find no "
        "further external knowledge needed, directly provide the answer inside <answer> and "
        "</answer>, without detailed illustrations. For example, <answer> Beijing </answer>. "
        "Question: "
    ),
    "thinkingtag": (
        "Answer the given question. You must conduct reasoning inside <thinking> and </thinking> "
        "first every time you get new information. After reasoning, if you find you lack some "
        "knowledge, you can call the search tool that is available to you; its results will be "
        "returned in a tool response. You can search as many times as you want. If you find no "
        "further external knowledge needed, directly provide the answer inside <answer> and "
        "</answer>, without detailed illustrations. For example, <answer> Beijing </answer>. "
        "Question: "
    ),
    "icl3hop": (
        "Answer the given question. You must conduct reasoning inside <thinking> and </thinking> "
        "first every time you get new information. Each <thinking> should be ONE BRIEF SENTENCE "
        "stating what you need next — not detailed analysis. After reasoning, if you lack "
        "knowledge, call the search tool. When you have enough information, provide the answer "
        "inside <answer> and </answer> without detailed illustrations.\n\n"
        "Here is an example showing the expected format and brevity:\n\n"
        "Question: What is the population of the capital city of the country where the inventor "
        "of the World Wide Web was born?\n\n"
        "<thinking>I need to find who invented the World Wide Web.</thinking>\n"
        '<tool_call>{"name": "search", "arguments": {"query_list": ["who invented the World Wide Web"]}}</tool_call>\n'
        '<tool_response>Doc 1 (Title: "Tim Berners-Lee"): Sir Timothy John Berners-Lee, also '
        "known as TimBL, is an English computer scientist best known as the inventor of the World "
        "Wide Web.</tool_response>\n"
        "<thinking>Tim Berners-Lee is English, so the country is the United Kingdom. I need the "
        "capital of the UK.</thinking>\n"
        '<tool_call>{"name": "search", "arguments": {"query_list": ["capital of the United Kingdom"]}}</tool_call>\n'
        '<tool_response>Doc 1 (Title: "London"): London is the capital and largest city of England '
        "and the United Kingdom.</tool_response>\n"
        "<thinking>The capital is London. Now I need the population of London.</thinking>\n"
        '<tool_call>{"name": "search", "arguments": {"query_list": ["population of London"]}}</tool_call>\n'
        '<tool_response>Doc 1 (Title: "London"): London has a population of approximately 9 '
        "million people.</tool_response>\n"
        "<thinking>The population is about 9 million.</thinking>\n"
        "<answer>9 million</answer>\n\nNow answer this question: "
    ),
}


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


def parse_tool_call(text: str) -> dict[str, Any] | None:
    """Return the first structurally valid Hermes tool call."""
    for candidate in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(value, dict)
            and isinstance(value.get("name"), str)
            and "arguments" in value
        ):
            return value
    return None


def _passages_to_string(passages: list[dict[str, Any]]) -> str:
    rendered = []
    for index, item in enumerate(passages):
        contents = str(item["document"]["contents"])
        title, _, body = contents.partition("\n")
        rendered.append(f"Doc {index + 1} (Title: {title})\n{body}".rstrip())
    return "\n\n".join(rendered)


def _render_passages(response: dict[str, Any]) -> str:
    results = response.get("result", [])
    if not results:
        return "No search results found."
    return "\n---\n".join(_passages_to_string(passages) for passages in results)


def format_search_response(response: dict[str, Any], official: bool = False) -> str:
    """Render retrieval results in the protocol the model expects.

    official=True reproduces upstream Search-R1's plain
    "\\n<information>...</information>\\n\\n" continuation. Otherwise the result
    is a JSON tool payload for the Hermes tool_call schema.
    """
    result = _render_passages(response)
    if official:
        return f"\n<information>{result}</information>\n\n"
    return json.dumps({"result": result}, ensure_ascii=False)


def truncate_tool_response(text: str, limit: int, side: str) -> str:
    if len(text) <= limit:
        return text
    if side == "left":
        return text[:limit] + "...(truncated)"
    if side == "right":
        return "(truncated)..." + text[-limit:]
    half = limit // 2
    return text[:half] + "...(truncated)..." + text[-half:]


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
        prompt_variant: str = "default",
        enable_thinking: bool = False,
        top_p: float = 0.95,
        sampling_top_k: int = -1,
        max_model_len: int = 3072,
        max_tool_response_length: int = 512,
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
        if prompt_variant not in PROMPT_VARIANTS:
            raise ValueError(f"unknown Search-R1 prompt variant: {prompt_variant}")

        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path)
        if engine is None or sampling_params_cls is None:
            from vllm import LLM, SamplingParams

            sampling_params_cls = sampling_params_cls or SamplingParams
            if engine is None:
                engine_kwargs = {
                    "model": model_path,
                    "tokenizer": model_path,
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

        # Which tool-calling protocol this variant speaks. The released
        # Search-R1 checkpoints were RL-trained on the <search> protocol.
        self.official_protocol = prompt_variant in OFFICIAL_PROTOCOL_VARIANTS
        self.model_path = model_path
        self.retrieval_url = retrieval_url
        self.max_steps = max_steps
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_timeout = retrieval_timeout
        self.retrieval_workers = retrieval_workers
        self.prompt_variant = prompt_variant
        self.enable_thinking = enable_thinking
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
        question = question.strip()
        if question and question[-1] != "?":
            question += "?"
        content = PROMPT_VARIANTS[self.prompt_variant] + question
        if self.official_protocol:
            # Upstream passes a single user turn with no system prompt and no
            # tool schema; the protocol lives entirely in the prompt text.
            return self.tok.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        return self.tok.apply_chat_template(
            messages,
            tools=[SEARCH_TOOL_SCHEMA],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    def _tool_continuation(self, result: str) -> str:
        if self.official_protocol:
            # Already wrapped in <information>...</information>; appending it
            # verbatim keeps the trajectory a single assistant turn, which is
            # what the policy saw during training.
            return result
        return self.tok.apply_chat_template(
            [{"role": "tool", "content": result}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
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
                        official=self.official_protocol,
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
        error = json.dumps({"result": f"Search error: {last_error}"}, ensure_ascii=False)
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
                params_kwargs = {
                    "n": 1,
                    "temperature": temperature,
                    "top_p": self.top_p,
                    "top_k": self.sampling_top_k,
                    "max_tokens": budget,
                }
                if self.official_protocol:
                    # Upstream stops generation at </search> so the query can be
                    # executed and the <information> block spliced in. Without
                    # this the model keeps going and hallucinates its own
                    # search results instead of waiting for real ones.
                    params_kwargs["stop"] = ["</search>"]
                    params_kwargs["include_stop_str_in_output"] = True
                sampling_params.append(self._sampling_params_cls(**params_kwargs))

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
                prompts[index] += generated
                assistant_text[index] += generated
                response_tokens[index] += len(generated_ids)
                if self.official_protocol:
                    search_query = parse_search_query(generated)
                    tool_call = (
                        {"name": "search", "arguments": {"query_list": [search_query]}}
                        if search_query
                        else None
                    )
                else:
                    tool_call = parse_tool_call(generated)

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

                arguments = tool_call.get("arguments")
                query_list = arguments.get("query_list") if isinstance(arguments, dict) else None
                step = AgentStep(
                    prompt=prompt_snapshots[index],
                    answer=generated,
                    action="toolcall",
                    tool_name=tool_call["name"],
                    tool_args=arguments if isinstance(arguments, dict) else None,
                )
                if tool_call["name"] != "search":
                    step.error = f"unknown tool: {tool_call['name']}"
                    query_list = None
                elif not isinstance(query_list, list) or not query_list:
                    step.error = "search requires a non-empty query_list"
                    query_list = None

                if query_list is None:
                    result = truncate_tool_response(
                        json.dumps({"result": f"Search error: {step.error}"}),
                        self.max_tool_response_length,
                        self.tool_response_truncate_side,
                    )
                    step.tool_result = result
                    pending.append((index, [], step))
                else:
                    pending.append((index, query_list, step))

            retrieval_inputs = [queries for _, queries, _ in pending if queries]
            retrieval_results = iter(self._retrieve_many(retrieval_inputs))
            for index, tool_queries, step in pending:
                result = next(retrieval_results) if tool_queries else step.tool_result
                step.tool_result = result
                continuation = self._tool_continuation(result)
                continuation_ids = self._token_ids(continuation)
                if (
                    response_tokens[index] + len(continuation_ids) >= response_limit
                    or len(self._token_ids(prompts[index])) + len(continuation_ids)
                    >= self.max_model_len
                ):
                    step.error = step.error or "tool response exceeded trajectory budget"
                    finished[index] = True
                else:
                    prompts[index] += continuation
                    response_tokens[index] += len(continuation_ids)
                traces[index].append(step)

        answers = [extract_answer(text) for text in assistant_text]
        return answers, traces
