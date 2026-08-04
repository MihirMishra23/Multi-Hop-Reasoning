"""Interleaved Retrieval with Chain-of-Thought (IRCoT) agent.

This is a lightweight adaptation of the official IRCoT inference loop for the
repository's local LLM and BM25 interfaces.  It preserves the defining
behavior: retrieve from the question, generate one reasoning sentence,
retrieve from the latest factual sentence, and accumulate evidence until an
answer is reached or the step budget is exhausted.

Reference implementation:
https://github.com/StonyBrookNLP/ircot/tree/3c1820f698eea5eeddb4fba3c56b64c961e063e4
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from agent.agent_class import Agent, AgentStep, LLM
from tools.retrieval import (
    BaseRetriever,
    FlashRAGBM25CorpusRetriever,
    FlashRAGBM25Retriever,
)


OFFICIAL_IRCOT_COMMIT = "3c1820f698eea5eeddb4fba3c56b64c961e063e4"
OFFICIAL_IRCOT_URL = "https://github.com/StonyBrookNLP/ircot"

_ANSWER_PATTERNS = (
    re.compile(r"FINAL_ANSWER\s*:\s*(.+)", re.IGNORECASE | re.DOTALL),
    re.compile(r"(?:the\s+)?answer\s+is\s*:?\s*(.+?)(?:\.|$)", re.IGNORECASE | re.DOTALL),
)
_REASONING_STARTERS = ("thus", "so", "that is", "therefore", "hence")
_WH_WORDS = {"who", "what", "when", "where", "why", "which", "how", "does", "is"}


def _doc_to_text(doc: Any) -> str:
    if isinstance(doc, dict):
        title = str(doc.get("title", "")).strip()
        contents = doc.get("contents")
        if contents is None:
            contents = doc.get("context")
        if contents is None:
            contents = doc.get("paragraph_text", "")
        body = str(contents).strip()
        if title and body:
            return f"Wikipedia Title: {title}\n{body}"
        return body or title
    return str(doc).strip()


def _document_key(doc: Any) -> Tuple[str, str]:
    if isinstance(doc, dict):
        title = str(doc.get("title", "")).strip().casefold()
        contents = doc.get("contents")
        if contents is None:
            contents = doc.get("context")
        if contents is None:
            contents = doc.get("paragraph_text", "")
        return title, re.sub(r"\s+", " ", str(contents)).strip().casefold()
    return "", re.sub(r"\s+", " ", str(doc)).strip().casefold()


def _first_sentence(text: str) -> str:
    """Return one generated reasoning sentence without requiring spaCy."""
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return ""
    answer_match = _extract_answer(cleaned)
    if answer_match is not None:
        return cleaned
    match = re.match(r"^(.+?[.!?])(?:\s|$)", cleaned)
    return match.group(1).strip() if match else cleaned


def _extract_answer(text: str) -> Optional[str]:
    for pattern in _ANSWER_PATTERNS:
        match = pattern.search(text or "")
        if match:
            answer = match.group(1).strip()
            answer = re.split(r"\n", answer, maxsplit=1)[0].strip()
            return answer.rstrip(".").strip()
    return None


def _is_reasoning_sentence(sentence: str) -> bool:
    lowered = (sentence or "").strip().casefold()
    return any(
        lowered == starter
        or lowered.startswith(f"{starter} ")
        or lowered.startswith(f"{starter},")
        for starter in _REASONING_STARTERS
    )


def _retrieval_query(question: str, thoughts: List[str]) -> str:
    factual_thoughts = [thought for thought in thoughts if not _is_reasoning_sentence(thought)]
    source = factual_thoughts[-1] if factual_thoughts else question
    words = [word for word in source.split() if word.casefold().strip(".,?!:;") not in _WH_WORDS]
    query = " ".join(words).strip()
    return query or question


class IRCoTAgent(Agent):
    """Interleave cumulative BM25 retrieval with sentence-level reasoning."""

    def __init__(
        self,
        llm: LLM,
        retriever_type: str = "bm25",
        contexts: Optional[List[Any]] = None,
        corpus: Optional[List[Any]] = None,
        retrieval_k: int = 6,
        max_evidence: int = 15,
        max_steps: int = 8,
        step_max_tokens: int = 96,
        retriever: Optional[BaseRetriever] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(llm=llm, max_steps=max_steps)
        if retrieval_k <= 0:
            raise ValueError("IRCoT retrieval_k must be positive")
        if max_evidence <= 0:
            raise ValueError("IRCoT max_evidence must be positive")
        if step_max_tokens <= 0:
            raise ValueError("IRCoT step_max_tokens must be positive")
        if (retriever_type or "").casefold() != "bm25":
            raise NotImplementedError("IRCoT currently supports only BM25 retrieval")

        self.contexts = list(contexts or [])
        self._corpus = list(corpus or [])
        self.retrieval_k = retrieval_k
        self.max_evidence = max_evidence
        self.step_max_tokens = step_max_tokens
        self.retriever: BaseRetriever = retriever or (
            FlashRAGBM25CorpusRetriever(self._corpus)
            if self._corpus
            else FlashRAGBM25Retriever()
        )
        self._evidence_docs: List[Any] = []
        self._retrieval_rounds: List[Dict[str, Any]] = []

    def reset(self, contexts: Optional[List[Any]] = None) -> None:
        if not self._corpus:
            self.contexts = list(contexts or [])
        self.trace = []
        self._evidence_docs = []
        self._retrieval_rounds = []

    def _retrieve(self, query: str, round_index: int) -> None:
        documents = self._corpus if self._corpus else self.contexts
        retrieved = self.retriever.retrieve(
            query=query,
            documents=documents,
            top_k=self.retrieval_k,
        )

        seen = {_document_key(doc) for doc in self._evidence_docs}
        added: List[Any] = []
        for doc in retrieved:
            key = _document_key(doc)
            if not key[1] or key in seen:
                continue
            if len(self._evidence_docs) >= self.max_evidence:
                break
            seen.add(key)
            self._evidence_docs.append(doc)
            added.append(doc)

        self._retrieval_rounds.append(
            {
                "round": round_index,
                "query": query,
                "retrieved": [_doc_to_text(doc) for doc in retrieved],
                "added": [_doc_to_text(doc) for doc in added],
                "cumulative_evidence_count": len(self._evidence_docs),
            }
        )

    def _reasoning_prompt(self, question: str, thoughts: List[str]) -> str:
        evidence = "\n\n".join(_doc_to_text(doc) for doc in self._evidence_docs)
        reasoning = " ".join(thoughts).strip()
        return (
            "Use the retrieved evidence to solve the multi-step question. Generate exactly one "
            "next reasoning sentence. Do not repeat previous reasoning. If the answer is now "
            "known, write `FINAL_ANSWER: <answer>`.\n\n"
            f"Retrieved evidence:\n{evidence}\n\n"
            f"Question: {question}\n"
            f"Reasoning so far: {reasoning}\n"
            "Next sentence:"
        )

    def _answer_prompt(self, question: str, thoughts: List[str]) -> str:
        evidence = "\n\n".join(_doc_to_text(doc) for doc in self._evidence_docs)
        reasoning = " ".join(thoughts).strip()
        return (
            "Answer the question using the retrieved evidence and reasoning. Return only the "
            "short answer prefixed by `FINAL_ANSWER:`.\n\n"
            f"Retrieved evidence:\n{evidence}\n\n"
            f"Reasoning:\n{reasoning}\n\n"
            f"Question: {question}"
        )

    def run(
        self,
        question: List[str] | str,
        **llm_kwargs: Any,
    ) -> Tuple[List[Optional[str]], List[List[AgentStep]]]:
        if isinstance(question, list):
            if len(question) != 1:
                raise NotImplementedError("IRCoT requires batch size 1")
            question = question[0]

        self.reset(self.contexts)
        thoughts: List[str] = []
        answer_hint: Optional[str] = None
        max_tokens = llm_kwargs.get("max_tokens")
        step_kwargs = dict(llm_kwargs)
        step_kwargs["max_tokens"] = min(max_tokens or self.step_max_tokens, self.step_max_tokens)

        for round_index in range(self.max_steps):
            query = _retrieval_query(question, thoughts)
            self._retrieve(query, round_index)

            prompt = self._reasoning_prompt(question, thoughts)
            try:
                response = self.llm.run(prompt, **step_kwargs)
                sentence = _first_sentence(response.text)
                answer_hint = _extract_answer(sentence)
                step = AgentStep(
                    prompt=prompt,
                    answer=sentence,
                    action="finish" if answer_hint is not None else "generate",
                )
            except Exception as exc:
                step = AgentStep(prompt=prompt, answer=None, action="finish", error=str(exc))
                self.trace.append(step)
                break

            self.trace.append(step)
            if sentence:
                thoughts.append(sentence)
            if answer_hint is not None or not sentence:
                break

        final_prompt = self._answer_prompt(question, thoughts)
        final_answer = answer_hint
        try:
            response = self.llm.run(final_prompt, **llm_kwargs)
            parsed = _extract_answer(response.text)
            if parsed is not None:
                final_answer = parsed
            elif response.text.strip():
                final_answer = response.text.strip()
            self.trace.append(
                AgentStep(prompt=final_prompt, answer=final_answer, action="finish")
            )
        except Exception as exc:
            self.trace.append(
                AgentStep(prompt=final_prompt, answer=final_answer, action="finish", error=str(exc))
            )

        return [final_answer], [list(self.trace)]
