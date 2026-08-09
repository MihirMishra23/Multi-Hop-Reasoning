"""Faithful IRCoT inference loop with a repository-local LLM adapter.

The control flow and prompt construction in this module are ported from the
official IRCoT implementation pinned by ``OFFICIAL_IRCOT_COMMIT``.  The model
call itself is intentionally supplied by this repository (for example Qwen),
so results are not the retired ``code-davinci-002`` results from the paper.

Upstream: https://github.com/StonyBrookNLP/ircot
License: Apache-2.0 (see ``third_party/ircot/NOTICE.md``)
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import time
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from rapidfuzz import fuzz

from agent.agent_class import Agent, AgentStep, LLM
from agent.ircot_constants import (
    OFFICIAL_CORPUS_NAMES,
    OFFICIAL_IRCOT_COMMIT,
    OFFICIAL_IRCOT_URL,
)


OFFICIAL_GENERATOR_MAX_TOKENS = 300
OFFICIAL_MODEL_LENGTH_LIMIT = 8000
OFFICIAL_MAX_NUM_SENTENCES = 10
OFFICIAL_MAX_NUM_PARAS = 15

random.seed(100)  # Matches the official module-level seed.

_ANSWER_REGEX = re.compile(r".* answer is:? (.*)\.?")
_ARITHMETIC_REASONING_REGEX = re.compile(
    r"(.*)(\d[\d,]*\.?\d+|\d+) ([+-]) (\d[\d,]*\.?\d+|\d+) = "
    r"(\d[\d,]*\.?\d+|\d+)(.*)"
)
_REASONING_STARTERS = ["thus ", "thus,", "so ", "so,", "that is,", "therefore", "hence"]
_WH_WORDS = {"who", "what", "when", "where", "why", "which", "how", "does", "is"}

# Exact 15-demonstration prompt sets from upstream ``run.py``.  The paper tunes
# hyperparameters using prompt set 1, then reuses the selected hyperparameters
# for prompt sets 2 and 3 on the test set.
OFFICIAL_PROMPT_QIDS: Dict[str, Dict[str, List[str]]] = {
    "hotpotqa": {
        "1": [
            "5abb14bd5542992ccd8e7f07", "5ac2ada5554299657fa2900d",
            "5a758ea55542992db9473680", "5ae0185b55429942ec259c1b",
            "5a8ed9f355429917b4a5bddd", "5abfb3435542990832d3a1c1",
            "5ab92dba554299131ca422a2", "5a835abe5542996488c2e426",
            "5a89c14f5542993b751ca98a", "5a90620755429933b8a20508",
            "5a7bbc50554299042af8f7d0", "5a8f44ab5542992414482a25",
            "5add363c5542990dbb2f7dc8", "5a7fc53555429969796c1b55",
            "5a790e7855429970f5fffe3d",
        ],
        "2": [
            "5a90620755429933b8a20508", "5a88f9d55542995153361218",
            "5a758ea55542992db9473680", "5a89c14f5542993b751ca98a",
            "5abfb3435542990832d3a1c1", "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76", "5a7fc53555429969796c1b55",
            "5a8f44ab5542992414482a25", "5a835abe5542996488c2e426",
            "5ac2ada5554299657fa2900d", "5a8ed9f355429917b4a5bddd",
            "5a754ab35542993748c89819", "5add363c5542990dbb2f7dc8",
            "5abb14bd5542992ccd8e7f07",
        ],
        "3": [
            "5a89d58755429946c8d6e9d9", "5a758ea55542992db9473680",
            "5a7fc53555429969796c1b55", "5a7bbc50554299042af8f7d0",
            "5a77acab5542992a6e59df76", "5a90620755429933b8a20508",
            "5a89c14f5542993b751ca98a", "5ab92dba554299131ca422a2",
            "5a8f44ab5542992414482a25", "5ae0185b55429942ec259c1b",
            "5a835abe5542996488c2e426", "5a754ab35542993748c89819",
            "5ac2ada5554299657fa2900d", "5a790e7855429970f5fffe3d",
            "5adfad0c554299603e41835a",
        ],
    },
    "2wiki": {
        "1": [
            "228546780bdd11eba7f7acde48001122", "97954d9408b011ebbd84ac1f6bf848b6",
            "a5995da508ab11ebbd82ac1f6bf848b6", "1ceeab380baf11ebab90acde48001122",
            "35bf3490096d11ebbdafac1f6bf848b6", "f86b4a28091711ebbdaeac1f6bf848b6",
            "f44939100bda11eba7f7acde48001122", "e5150a5a0bda11eba7f7acde48001122",
            "c6805b2908a911ebbd80ac1f6bf848b6", "13cda43c09b311ebbdb0ac1f6bf848b6",
            "f1ccdfee094011ebbdaeac1f6bf848b6", "028eaef60bdb11eba7f7acde48001122",
            "8727d1280bdc11eba7f7acde48001122", "79a863dc0bdc11eba7f7acde48001122",
            "c6f63bfb089e11ebbd78ac1f6bf848b6",
        ],
        "2": [
            "c6805b2908a911ebbd80ac1f6bf848b6", "5897ec7a086c11ebbd61ac1f6bf848b6",
            "028eaef60bdb11eba7f7acde48001122", "af8c6722088b11ebbd6fac1f6bf848b6",
            "1ceeab380baf11ebab90acde48001122", "5811079c0bdc11eba7f7acde48001122",
            "228546780bdd11eba7f7acde48001122", "e5150a5a0bda11eba7f7acde48001122",
            "f44939100bda11eba7f7acde48001122", "f1ccdfee094011ebbdaeac1f6bf848b6",
            "13cda43c09b311ebbdb0ac1f6bf848b6", "79a863dc0bdc11eba7f7acde48001122",
            "a5995da508ab11ebbd82ac1f6bf848b6", "cdbb82ec0baf11ebab90acde48001122",
            "c6f63bfb089e11ebbd78ac1f6bf848b6",
        ],
        "3": [
            "028eaef60bdb11eba7f7acde48001122", "8727d1280bdc11eba7f7acde48001122",
            "79a863dc0bdc11eba7f7acde48001122", "4724c54e08e011ebbda1ac1f6bf848b6",
            "e5150a5a0bda11eba7f7acde48001122", "35bf3490096d11ebbdafac1f6bf848b6",
            "a5995da508ab11ebbd82ac1f6bf848b6", "228546780bdd11eba7f7acde48001122",
            "97954d9408b011ebbd84ac1f6bf848b6", "f44939100bda11eba7f7acde48001122",
            "1ceeab380baf11ebab90acde48001122", "f86b4a28091711ebbdaeac1f6bf848b6",
            "c6f63bfb089e11ebbd78ac1f6bf848b6", "af8c6722088b11ebbd6fac1f6bf848b6",
            "5897ec7a086c11ebbd61ac1f6bf848b6",
        ],
    },
    "musique": {
        "1": [
            "2hop__804754_52230", "2hop__292995_8796", "2hop__496817_701819",
            "2hop__154225_727337", "2hop__642271_608104", "2hop__439265_539716",
            "2hop__195347_20661", "2hop__131516_53573", "2hop__427213_79175",
            "3hop1__443556_763924_573834", "2hop__782642_52667",
            "2hop__861128_15822", "4hop3__703974_789671_24078_24137",
            "3hop1__61746_67065_43617", "4hop3__463724_100414_35260_54090",
        ],
        "2": [
            "2hop__292995_8796", "2hop__154225_727337", "2hop__642271_608104",
            "2hop__195347_20661", "3hop1__61746_67065_43617", "2hop__861128_15822",
            "3hop1__753524_742157_573834", "2hop__496817_701819",
            "4hop3__703974_789671_24078_24137", "3hop1__858730_386977_851569",
            "2hop__804754_52230", "2hop__782642_52667", "2hop__102217_58400",
            "2hop__387702_20661", "3hop1__443556_763924_573834",
        ],
        "3": [
            "2hop__427213_79175", "3hop1__753524_742157_573834",
            "2hop__782642_52667", "2hop__496817_701819",
            "3hop1__443556_763924_573834", "4hop3__463724_100414_35260_54090",
            "2hop__292995_8796", "2hop__804754_52230",
            "3hop1__858730_386977_851569", "2hop__131516_53573",
            "2hop__387702_20661", "4hop3__703974_789671_24078_24137",
            "2hop__154225_727337", "3hop1__61746_67065_43617",
            "2hop__642271_608104",
        ],
    },
}

OFFICIAL_PROMPT_HASHES = {
    "00fd7b411360004f55c5d295d71d257d45f00c8125033faa88232935607993c3",
    "af56b51582a5fdbebf5a5d50ab95cefcb95d92986cf820774ccbaa897a71552f",
    "299fbb3b9e3b642cbe4c09ec035e64a2627a5d118aa2aaede8c8829e4ab7dbe8",
    "890050a6dd6c396b2af3cb2c1f88f01049e7bee49e64f11cb5fd08eaae27015a",
    "6cd153370f5f35a0e3db53d8def28abfe341ae67cddd2ccc5365eeb61769e235",
    "f4ffc3fdcf1616b60b5317fc44da61ba1306c8413d86c59d01621b0433daad95",
    "4f6b4b337de812ff94dbc713e3eabdbeb97f93fd5549ac2e3b92c5e0813cf5d0",
    "454bac330eb8800e629a2ca9938b7358da8131a75c563b51ff1836cf4501377f",
    "406602227e58733de4396230008ea9ccc8a2c167be2006f24e4fe77e3446b5d5",
}


@lru_cache(maxsize=15)
def _get_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return AutoTokenizer.from_pretrained(model_name)


def _token_length(text: str, tokenizer_model_name: str = "gpt2") -> int:
    return len(_get_tokenizer(tokenizer_model_name).tokenize(text))


def read_official_prompt(
    file_path: str,
    dataset: str,
    prompt_set: str = "1",
    estimated_generation_length: int = OFFICIAL_GENERATOR_MAX_TOKENS,
    model_length_limit: int = OFFICIAL_MODEL_LENGTH_LIMIT,
    tokenizer_model_name: str = "gpt2",
) -> str:
    """Port of the official prompt reader for the released Codex prompts."""
    if dataset not in OFFICIAL_PROMPT_QIDS:
        raise ValueError(f"Official IRCoT prompts are unavailable for dataset={dataset!r}")
    if prompt_set not in OFFICIAL_PROMPT_QIDS[dataset]:
        raise ValueError(f"Official IRCoT prompt_set must be 1, 2, or 3; got {prompt_set!r}")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"IRCoT prompt file not found: {file_path}. Run scripts/fetch_ircot_assets.py."
        )

    with open(file_path, "r", encoding="utf-8") as prompt_file:
        all_prompt_lines = [line + "\n" for line in prompt_file.read().strip().split("\n")]

    metadata_prefix = "# METADATA: "
    example: Dict[str, Any] = {"default": True, "lines": []}
    examples: List[Dict[str, Any]] = []
    for index, line in enumerate(all_prompt_lines):
        if index == len(all_prompt_lines) - 1:
            examples.append(example)
        if line.strip().startswith(metadata_prefix):
            examples.append(example)
            metadata = json.loads(line.strip().replace(metadata_prefix, "", 1))
            example = copy.deepcopy(metadata)
            example["lines"] = []
        else:
            example["lines"].append(line)

    qids = OFFICIAL_PROMPT_QIDS[dataset][prompt_set]
    examples = [example for example in examples if example.get("qid") in qids and example["lines"]]
    examples = sorted(examples, key=lambda item: qids.index(item["qid"]))
    texts = ["".join(example["lines"]).strip() for example in examples]
    if not texts:
        raise ValueError(f"No official demonstrations were found in {file_path}")
    if len(texts) == 1:
        return texts[0].strip()

    lengths = [_token_length(text, tokenizer_model_name) for text in texts]
    while lengths:
        estimated_total = sum(lengths) + max(lengths) + estimated_generation_length
        if estimated_total <= model_length_limit:
            break
        texts.pop()
        lengths.pop()
    if not texts:
        raise ValueError("Official prompt compression removed every demonstration")
    return "\n\n\n".join(text.strip() for text in texts).strip()


def fit_prompt_into_official_limit(
    prompt: str,
    estimated_generation_length: int = OFFICIAL_GENERATOR_MAX_TOKENS,
    model_length_limit: int = OFFICIAL_MODEL_LENGTH_LIMIT,
    tokenizer_model_name: str = "gpt2",
) -> str:
    """Port of GPT3Generator's first-demonstration-first prompt fitting."""
    demonstrations = [part.strip() for part in prompt.strip().split("\n\n\n") if part.strip()]
    if not demonstrations:
        return ""
    test_example = demonstrations.pop()
    test_size = _token_length(test_example, tokenizer_model_name)
    sizes = [_token_length(item, tokenizer_model_name) for item in demonstrations]
    while demonstrations and sum(sizes) + test_size + estimated_generation_length >= model_length_limit:
        demonstrations.pop(0)
        sizes.pop(0)
    updated = "\n\n\n".join(demonstrations + [test_example])
    if _token_length(updated, tokenizer_model_name) + estimated_generation_length > model_length_limit:
        lines = updated.split("\n")
        while lines:
            lines.pop(0)
            if _token_length("\n".join(lines), tokenizer_model_name) <= model_length_limit:
                break
        updated = "\n".join(lines)
    return updated


def is_reasoning_sentence(sentence: str) -> bool:
    for starter in _REASONING_STARTERS:
        if sentence.lower().startswith(starter):
            return True
    return bool(re.match(_ARITHMETIC_REASONING_REGEX, sentence))


def remove_reasoning_sentences(sentences: Sequence[str]) -> List[str]:
    return [sentence for sentence in sentences if not is_reasoning_sentence(sentence)]


def remove_wh_words(text: str) -> str:
    words = [word for word in text.split(" ") if word.strip().lower() not in _WH_WORDS]
    return " ".join(words)


def retrieval_query(question: str, generated_sentences: Sequence[str]) -> str:
    factual = remove_reasoning_sentences(generated_sentences)
    latest = factual[-1].strip() if factual else ""
    return remove_wh_words(latest if latest else question)


def para_to_text(title: str, paragraph: str, max_num_words: int = 350) -> str:
    paragraph = " ".join(paragraph.split(" ")[:max_num_words])
    if paragraph.strip().startswith("Wikipedia Title: "):
        return paragraph.strip()
    return "Wikipedia Title: " + title + "\n" + paragraph.strip()


def is_para_closely_matching(
    existing_titles: Sequence[str],
    existing_paras: Sequence[str],
    new_title: str,
    new_para: str,
    match_threshold: float = 90,
) -> bool:
    if new_title in existing_titles and new_para in existing_paras:
        return True
    for title, paragraph in zip(existing_titles, existing_paras):
        if fuzz.ratio(title, new_title) >= match_threshold and fuzz.ratio(paragraph, new_para) >= match_threshold:
            return True
    return False


def extract_official_answer(text: str) -> str:
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    match = _ANSWER_REGEX.match(text)
    answer = match.group(1) if match else text
    if answer.endswith("."):
        answer = answer[:-1]
    return answer


@lru_cache(maxsize=1)
def _spacy_sentence_segmenter():
    import spacy

    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError(
            "Faithful IRCoT requires spaCy en_core_web_sm for sentence boundaries; "
            "run `python -m spacy download en_core_web_sm`."
        ) from exc


def first_spacy_sentence(text: str) -> str:
    sentences = list(_spacy_sentence_segmenter()(text.strip()).sents)
    return sentences[0].text if sentences else ""


class OfficialIRCoTRetriever:
    """Client for the HTTP retrieval endpoint released with IRCoT."""

    def __init__(
        self,
        endpoint: str,
        corpus_name: str,
        request_timeout: Optional[float] = None,
        retry_count: int = 10,
        retry_delay_seconds: float = 20,
    ) -> None:
        if not endpoint:
            raise ValueError("Official IRCoT retrieval requires --ircot-retriever-url")
        self.endpoint = endpoint.rstrip("/")
        if not self.endpoint.endswith("/retrieve"):
            self.endpoint += "/retrieve"
        self.corpus_name = corpus_name
        self.request_timeout = request_timeout
        self.retry_count = retry_count
        self.retry_delay_seconds = retry_delay_seconds

    def retrieve(self, query: str, documents: Iterable[Any], top_k: int) -> List[Dict[str, Any]]:
        del documents
        params = {
            "retrieval_method": "retrieve_from_elasticsearch",
            "query_text": query,
            "max_hits_count": top_k,
            "corpus_name": self.corpus_name,
            "document_type": "title_paragraph_text",
        }
        last_error: Optional[Exception] = None
        for attempt in range(self.retry_count):
            try:
                response = requests.post(
                    self.endpoint,
                    json=params,
                    timeout=self.request_timeout,
                )
                if not response.ok:
                    raise IRCoTRetrievalResponseError(
                        f"IRCoT retrieval server returned HTTP {response.status_code}"
                    )
                payload = response.json()
                retrieval = payload["retrieval"]
                if not isinstance(retrieval, list):
                    raise TypeError("IRCoT retriever response field 'retrieval' is not a list")
                return retrieval
            except IRCoTRetrievalResponseError:
                raise
            except requests.RequestException as exc:
                last_error = exc
                if attempt + 1 < self.retry_count:
                    time.sleep(self.retry_delay_seconds)
        raise RuntimeError("IRCoT retrieval request failed after retries") from last_error


class IRCoTRetrievalResponseError(RuntimeError):
    """A non-success response, which official IRCoT skips up to nine times."""


class IRCoTAgent(Agent):
    """Official interleaved retrieval/reasoning algorithm with a substituted LLM."""

    def __init__(
        self,
        llm: LLM,
        dataset: str,
        prompt_file: str,
        retriever: Optional[Any] = None,
        retriever_url: Optional[str] = None,
        retrieval_k: int = 6,
        max_evidence: int = OFFICIAL_MAX_NUM_PARAS,
        max_steps: int = OFFICIAL_MAX_NUM_SENTENCES,
        generator_max_tokens: int = OFFICIAL_GENERATOR_MAX_TOKENS,
        prompt_set: str = "1",
        sentence_segmenter: Callable[[str], str] = first_spacy_sentence,
        verify_prompt_hash: bool = True,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if dataset not in OFFICIAL_CORPUS_NAMES:
            raise ValueError("Faithful IRCoT supports hotpotqa, 2wiki, and musique")
        if retrieval_k <= 0 or max_evidence <= 0 or max_steps <= 0:
            raise ValueError("IRCoT retrieval_k, max_evidence, and max_steps must be positive")
        super().__init__(llm=llm, max_steps=max_steps)
        self.dataset = dataset
        self.retrieval_k = retrieval_k
        self.max_evidence = max_evidence
        self.generator_max_tokens = generator_max_tokens
        self.prompt_set = str(prompt_set)
        self.sentence_segmenter = sentence_segmenter
        self.prompt_file = prompt_file
        with open(prompt_file, "rb") as source:
            self.prompt_sha256 = hashlib.sha256(source.read()).hexdigest()
        if verify_prompt_hash and self.prompt_sha256 not in OFFICIAL_PROMPT_HASHES:
            raise ValueError(
                f"IRCoT prompt checksum is not from pinned upstream commit: {self.prompt_sha256}"
            )
        self.prompt = read_official_prompt(prompt_file, dataset, prompt_set=self.prompt_set)
        self.retriever = retriever or OfficialIRCoTRetriever(
            endpoint=retriever_url or "",
            corpus_name=OFFICIAL_CORPUS_NAMES[dataset],
        )
        self.contexts: List[Any] = []
        self._evidence_docs: List[Dict[str, Any]] = []
        self._retrieval_rounds: List[Dict[str, Any]] = []
        self._retrieval_failures = 0

    def reset(self, contexts: Optional[List[Any]] = None) -> None:
        self.contexts = list(contexts or [])
        self.trace = []
        self._evidence_docs = []
        self._retrieval_rounds = []
        self._retrieval_failures = 0

    @staticmethod
    def _title_paragraph(document: Any) -> Tuple[str, str]:
        if not isinstance(document, dict):
            raise TypeError("Official IRCoT retrieval documents must be dictionaries")
        title = str(document.get("title", ""))
        paragraph = str(document.get("paragraph_text", document.get("contents", "")))
        return title, paragraph

    def _retrieve(self, query: str, round_index: int) -> None:
        retrieved: List[Dict[str, Any]] = []
        if query.strip():
            try:
                retrieved = self.retriever.retrieve(query=query, documents=[], top_k=self.retrieval_k)
            except IRCoTRetrievalResponseError:
                self._retrieval_failures += 1
                if self._retrieval_failures > 9:
                    raise

        existing_titles = [self._title_paragraph(doc)[0] for doc in self._evidence_docs]
        existing_paras = [self._title_paragraph(doc)[1] for doc in self._evidence_docs]
        added: List[Dict[str, Any]] = []
        for document in retrieved:
            title, paragraph = self._title_paragraph(document)
            corpus_name = document.get("corpus_name")
            expected_corpus = OFFICIAL_CORPUS_NAMES[self.dataset]
            if corpus_name is not None and corpus_name != expected_corpus:
                raise ValueError(f"Retrieved corpus {corpus_name!r} does not match {expected_corpus!r}")
            if len(paragraph.split(" ")) > 600:
                continue
            if is_para_closely_matching(existing_titles, existing_paras, title, paragraph):
                continue
            if len(self._evidence_docs) >= self.max_evidence:
                continue
            normalized = dict(document)
            normalized["title"] = title
            normalized["paragraph_text"] = paragraph
            self._evidence_docs.append(normalized)
            added.append(normalized)
            existing_titles.append(title)
            existing_paras.append(paragraph)

        self._retrieval_rounds.append(
            {
                "round": round_index,
                "query": query,
                "retrieved": retrieved,
                "added": added,
                "cumulative_evidence_count": len(self._evidence_docs),
            }
        )

    def _context(self) -> str:
        return "\n\n".join(
            para_to_text(*self._title_paragraph(document), max_num_words=350)
            for document in self._evidence_docs
        )

    def _reasoning_prompt(self, question: str, generated_sentences: Sequence[str]) -> str:
        test_example = self._context() + "\n\n" + f"Q: {question}\nA: {' '.join(generated_sentences)}"
        prompt = "\n\n\n".join([self.prompt, test_example]).strip()
        return fit_prompt_into_official_limit(prompt, self.generator_max_tokens)

    def _final_reader_prompt(self, question: str) -> str:
        prompt = self.prompt + "\n"
        context = self._context()
        if context:
            prompt += "\n\n" + context
        prompt += "\n\nQ: " + question + "\nA:"
        return fit_prompt_into_official_limit(prompt.rstrip(), self.generator_max_tokens)

    def _generate(self, prompt: str, llm_kwargs: Dict[str, Any]) -> str:
        generation_kwargs = dict(llm_kwargs)
        generation_kwargs["max_tokens"] = self.generator_max_tokens
        generation_kwargs["stop"] = ["\n"]
        extra = dict(generation_kwargs.get("extra") or {})
        extra["raw_prompt"] = True
        # The official prompt fitter guarantees an 8K-token input budget.
        # Override stale tokenizer metadata (Qwen snapshots may advertise
        # 1,024) so the final test question and retrieved evidence are not
        # silently truncated from the raw completion prompt.
        extra["max_input_tokens"] = OFFICIAL_MODEL_LENGTH_LIMIT
        generation_kwargs["extra"] = extra
        return self.llm.run(prompt, **generation_kwargs).text.strip()

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
        generated_sentences: List[str] = []
        for round_index in range(self.max_steps):
            query = retrieval_query(question, generated_sentences)
            self._retrieve(query, round_index)
            prompt = self._reasoning_prompt(question, generated_sentences)
            try:
                generation = self._generate(prompt, llm_kwargs)
                sentence = self.sentence_segmenter(generation)
                generated_sentences.append(sentence)
                should_exit = (
                    not sentence
                    or len(generated_sentences) >= self.max_steps
                    or bool(_ANSWER_REGEX.match(sentence))
                )
                self.trace.append(
                    AgentStep(
                        prompt=prompt,
                        answer=sentence,
                        action="finish" if should_exit else "generate",
                        raw_response=generation,
                    )
                )
                if should_exit:
                    break
            except Exception as exc:
                self.trace.append(AgentStep(prompt=prompt, answer=None, action="finish", error=str(exc)))
                break

        final_prompt = self._final_reader_prompt(question)
        final_answer: Optional[str] = None
        try:
            final_generation = self._generate(final_prompt, llm_kwargs)
            final_answer = extract_official_answer(final_generation)
            self.trace.append(
                AgentStep(
                    prompt=final_prompt,
                    answer=final_answer,
                    action="finish",
                    raw_response=final_generation,
                )
            )
        except Exception as exc:
            self.trace.append(AgentStep(prompt=final_prompt, answer=None, action="finish", error=str(exc)))
        return [final_answer], [list(self.trace)]
