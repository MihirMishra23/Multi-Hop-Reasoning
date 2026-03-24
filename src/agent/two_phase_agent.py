"""Two-phase LMLM agent for evaluation.

Phase 1: given wiki contexts, the model extracts knowledge triplets and we build
         a per-example DatabaseManager from them.
Phase 2: the model answers the question using DB lookups against the per-example DB.

This mirrors the `_generate_two_phase` logic in `trainer/lmlm_basetrainer.py` but
is stripped of all training machinery (no rollouts, no rewards, no gradients).
"""

import json
import os
import re
from typing import Optional

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from agent.agent_class import Agent, AgentStep
from multi_lmlm.database.database_manager import (
    DatabaseManager,
    build_databases_from_triplets_batch,
)
from multi_lmlm.constants import (
    ANSWER_END_TOKEN,
    ANSWER_START_TOKEN,
    DB_END_TOKEN,
    DB_RETRIEVE_TOKEN,
    DB_START_TOKEN,
)
from constants import REPO_ROOT


def _parse_triplets(text: str) -> list[tuple[str, str, str]]:
    """Parse tab-separated triplets from Phase 1 model output."""
    triplets: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        parts = line.strip().split("\t")
        if len(parts) == 3:
            triplets.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return triplets


class TwoPhaseAgent(Agent):
    """Evaluation agent that builds a per-question DB on-the-fly (no pre-built DB needed).

    Usage::

        agent = TwoPhaseAgent(model_path="...", phase1_prompt_type="sft")
        answers, traces = agent.run(queries, contexts=contexts_list)
    """

    def __init__(
        self,
        model_path: str,
        phase1_prompt_type: str = "sft",
        # retrieval
        top_k: int = 4,
        similarity_threshold: float = 0.6,
        use_inverses: bool = False,
        return_triplets: bool = False,
        # vLLM sampling (should match training)
        temperature: float = 1.0,
        top_p: float = 0.95,
        vllm_top_k: int = 4,          # vLLM sampling top_k, distinct from retrieval top_k
        repetition_penalty: float = 1.0,
        max_completion_length: int = 1024,
        max_model_len: int = 4096,
        **kwargs,
    ):
        self.model_path = model_path
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.use_inverses = use_inverses
        self.return_triplets = return_triplets
        # sampling params
        self.temperature = temperature
        self.top_p = top_p
        self.vllm_top_k = vllm_top_k
        self.repetition_penalty = repetition_penalty
        self.max_completion_length = max_completion_length

        # Phase 1 prompt template
        prompt_path = os.path.join(REPO_ROOT, "data", "prompts", "database_creation.json")
        prompt_key = "sft_with_question" if phase1_prompt_type == "with_question" else "sft"
        with open(prompt_path) as f:
            self._phase1_prompt_template: str = json.load(f)[prompt_key]["prompt"]
        self._phase1_prompt_type = phase1_prompt_type

        # Tokenizer + special token IDs
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self._stop_token_ids = [
            self.tok.eos_token_id,
            self.tok.encode(DB_RETRIEVE_TOKEN, add_special_tokens=False)[0],
        ]

        # Shared vLLM engine for both phases
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.6,
            seed=42,
            tokenizer=model_path,
            max_model_len=max_model_len,
        )

        self.max_turns = 16

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        queries: list[str],
        contexts: list[list[str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ):
        """Two-phase inference over a batch of questions.

        Args:
            queries: Questions to answer.
            contexts: Per-question context paragraphs (wiki passages).
            max_tokens: Max tokens per QA generation turn.
            temperature: Sampling temperature for Phase 2 QA.

        Returns:
            (answers, traces) — one entry per query.
        """
        assert len(queries) == len(contexts), (
            f"queries ({len(queries)}) and contexts ({len(contexts)}) must have the same length"
        )
        # Default to instance sampling params if not overridden
        _temperature = temperature if temperature is not None else self.temperature
        _max_tokens  = max_tokens  if max_tokens  is not None else self.max_completion_length

        per_example_dbs = self._phase1_build_dbs(queries, contexts)
        return self._phase2_qa(queries, per_example_dbs, _max_tokens, _temperature)

    # ------------------------------------------------------------------
    # Internal phases
    # ------------------------------------------------------------------

    def _phase1_build_dbs(
        self,
        queries: list[str],
        contexts: list[list[str]],
    ) -> list[DatabaseManager]:
        """Phase 1: generate triplets from contexts and build per-example DBs."""
        if self._phase1_prompt_type == "with_question":
            phase1_prompts = [
                self._phase1_prompt_template.format(
                    context="\n\n".join(ctx), question=q
                )
                for ctx, q in zip(contexts, queries)
            ]
        else:
            phase1_prompts = [
                self._phase1_prompt_template.format(context="\n\n".join(ctx))
                for ctx in contexts
            ]

        params = SamplingParams(
            n=1,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.vllm_top_k,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_completion_length,
            stop_token_ids=self._stop_token_ids,   # same as trainer: [EOS, DB_RETRIEVE_TOKEN]
        )
        outputs = self.llm.generate(phase1_prompts, params, use_tqdm=False)

        raw_texts = [out.outputs[0].text if out.outputs else "" for out in outputs]
        all_triplets = [_parse_triplets(t) for t in raw_texts]

        # Store for inspection / saving in eval
        self._phase1_info = [
            {
                "raw_text": raw,
                "triplets": triplets,
                "num_triplets": len(triplets),
                "num_context_paragraphs": len(ctx),
            }
            for raw, triplets, ctx in zip(raw_texts, all_triplets, contexts)
        ]

        dbs = build_databases_from_triplets_batch(
            all_triplets,
            top_k=self.top_k,
            default_threshold=self.similarity_threshold,
            adaptive=False,
            use_inverses=self.use_inverses,
        )
        self._phase1_dbs = dbs   # stored for external inspection (e.g. similarity scores)
        return dbs

    def _phase2_qa(
        self,
        queries: list[str],
        per_example_dbs: list[DatabaseManager],
        max_tokens: int,
        temperature: float,
    ):
        """Phase 2: answer questions with per-example DB lookups (multi-turn vLLM)."""
        B = len(queries)
        prompts = [f"Question:\n{q}\nAnswer:\n" for q in queries]
        active = [True] * B
        results: list[tuple[Optional[str], Optional[list]]] = [(None, None)] * B
        self._lookup_logs: list[list[dict]] = [[] for _ in range(B)]

        params = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=self.top_p,
            top_k=self.vllm_top_k,
            repetition_penalty=self.repetition_penalty,
            max_tokens=max_tokens,
            stop_token_ids=self._stop_token_ids,
        )

        max_model_len = self.llm.llm_engine.model_config.max_model_len

        for _turn in range(self.max_turns):
            active_indices = [i for i in range(B) if active[i]]
            if not active_indices:
                break

            # Deactivate any prompt whose length would exceed max_model_len
            for i in active_indices[:]:
                prompt_len = len(self.tok.encode(prompts[i], add_special_tokens=False))
                if prompt_len + max_tokens > max_model_len:
                    active[i] = False
                    active_indices.remove(i)

            if not active_indices:
                break

            outputs = self.llm.generate(
                [prompts[i] for i in active_indices], params, use_tqdm=False
            )

            for out, i in zip(outputs, active_indices):
                if out.outputs and out.outputs[0].token_ids:
                    generated = self.tok.decode(
                        out.outputs[0].token_ids, skip_special_tokens=False
                    )
                    prompts[i] += generated
                else:
                    generated = ""

                if ANSWER_END_TOKEN in prompts[i]:
                    active[i] = False
                    try:
                        answer = (
                            prompts[i]
                            .split(ANSWER_START_TOKEN)[1]
                            .split(ANSWER_END_TOKEN)[0]
                        )
                        results[i] = (answer, [AgentStep(prompts[i], answer, "generate")])
                    except Exception:
                        results[i] = ("", [AgentStep(prompts[i], "", "generate")])
                    continue

                if DB_RETRIEVE_TOKEN in generated:
                    return_value = "unknown"
                    log = {"query": None, "success": False, "returned_count": 0, "error": None}
                    try:
                        db_query = prompts[i].rsplit(DB_START_TOKEN)[-1]
                        log["query"] = db_query
                        values = per_example_dbs[i].retrieve_from_database(
                            DB_START_TOKEN + db_query,
                            threshold=self.similarity_threshold,
                            top_k=self.top_k,
                            return_triplets=self.return_triplets,
                        )
                        log["returned_count"] = len(values)
                        log["success"] = bool(values)
                        return_value = ", ".join(values)
                    except Exception as e:
                        log["error"] = str(e)
                    self._lookup_logs[i].append(log)
                    prompts[i] += return_value + DB_END_TOKEN

        # Finalize any queries that never emitted ANSWER_END_TOKEN
        for i in range(B):
            if results[i][0] is None:
                results[i] = ("", [AgentStep(prompts[i], "", "generate")])

        answers = [r[0] for r in results]
        traces = [r[1] for r in results]
        return answers, traces
