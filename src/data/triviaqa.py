"""Script for loading TriviaQA dataset.

This module exposes `load_triviaqa(setting, split, limit=None, seed=None)`
that returns a Hugging Face Dataset with a unified schema:

  - id: str
  - question: str
  - answers: List[str]
  - contexts: List[str]
  - supporting_facts: List[Dict[str, Any]]  # Empty for TriviaQA

Loading from Hugging Face datasets ("trivia_qa" with "rc" config).

TriviaQA uses the "rc" (reading comprehension) configuration which includes:
  - Entity pages with Wikipedia contexts
  - Search results with web search contexts
  - Multiple answer aliases for flexible evaluation
"""

import os
from typing import Any, Dict, List, Optional

from datasets import Dataset as HFDataset  # type: ignore
from datasets import load_dataset  # type: ignore


def _normalize_split(split: str) -> str:
    """Normalize split names to HuggingFace convention."""
    if split.lower() in {"dev", "validation"}:
        return "validation"
    return split.lower()


def _build_contexts_from_triviaqa(ex: Dict[str, Any]) -> List[str]:
    """
    Build context strings from TriviaQA's entity_pages and search_results.

    TriviaQA "rc" format provides:
      - entity_pages: dict with keys 'title' and 'wiki_context' (lists)
      - search_results: dict with keys 'title' and 'search_context' (lists)
    """
    contexts: List[str] = []

    # Extract from entity_pages (Wikipedia articles)
    entity_pages = ex.get("entity_pages", {})
    if isinstance(entity_pages, dict):
        titles = entity_pages.get("title", [])
        wiki_contexts = entity_pages.get("wiki_context", [])

        # Ensure they're lists
        if not isinstance(titles, list):
            titles = [titles] if titles else []
        if not isinstance(wiki_contexts, list):
            wiki_contexts = [wiki_contexts] if wiki_contexts else []

        for i, title in enumerate(titles):
            if i < len(wiki_contexts):
                context = wiki_contexts[i]
                if context and isinstance(context, str):
                    # Format: "Title: content"
                    paragraph = f"{title}: {context}".strip()
                    contexts.append(paragraph)

    # Extract from search_results (web search snippets)
    search_results = ex.get("search_results", {})
    if isinstance(search_results, dict):
        titles = search_results.get("title", [])
        search_contexts = search_results.get("search_context", [])

        # Ensure they're lists
        if not isinstance(titles, list):
            titles = [titles] if titles else []
        if not isinstance(search_contexts, list):
            search_contexts = [search_contexts] if search_contexts else []

        for i, title in enumerate(titles):
            if i < len(search_contexts):
                context = search_contexts[i]
                if context and isinstance(context, str):
                    # Format: "Title: content"
                    paragraph = f"{title}: {context}".strip()
                    contexts.append(paragraph)

    return contexts


def _build_answers_from_triviaqa(ex: Dict[str, Any]) -> List[str]:
    """
    Build answer list from TriviaQA's value and aliases fields.

    TriviaQA provides:
      - value: the primary answer string
      - aliases: list of alternative acceptable answers
      - normalized_aliases: normalized versions (we'll use regular aliases)
    """
    answers: List[str] = []

    # Add primary answer
    value = ex.get("value")
    if value and isinstance(value, str):
        answers.append(value)

    # Add aliases (alternative acceptable answers)
    aliases = ex.get("aliases", [])
    if isinstance(aliases, list):
        for alias in aliases:
            if alias and isinstance(alias, str) and alias not in answers:
                answers.append(alias)

    # Deduplicate while preserving order
    seen = set()
    unique_answers = []
    for ans in answers:
        ans_lower = ans.lower().strip()
        if ans_lower and ans_lower not in seen:
            seen.add(ans_lower)
            unique_answers.append(ans.strip())

    return unique_answers


def _normalize_triviaqa_dataset(ds: HFDataset) -> HFDataset:
    """Normalize TriviaQA dataset to unified schema."""
    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(ex.get("question_id", "")),
            "question": str(ex.get("question", "")),
            "answers": _build_answers_from_triviaqa(ex),
            "contexts": _build_contexts_from_triviaqa(ex),
            "supporting_facts": [],  # TriviaQA doesn't have explicit supporting facts
        }

    # Avoid cache conflicts in distributed settings
    return ds.map(
        _map,
        remove_columns=ds.column_names,
        desc="normalize triviaqa",
        load_from_cache_file=False,
        keep_in_memory=True,
    )


def load_triviaqa(
    setting: str = "rc",
    split: str = "train",
    limit: Optional[int] = None,
    seed: Optional[int] = None,
) -> HFDataset:
    """Load TriviaQA with unified schema.

    Args:
        setting: Dataset configuration. Options: "rc" (reading comprehension - recommended),
                 "rc.nocontext", "unfiltered", "unfiltered.nocontext".
                 Default is "rc" which includes contexts and filters for answerable questions.
        split: "train", "validation"/"dev", or "test".
        limit: optional max number of rows to return.
        seed: optional random seed for shuffling. If provided, dataset will be shuffled deterministically.

    Returns:
        HFDataset with schema:
            - id: str
            - question: str
            - answers: List[str]
            - contexts: List[str]
            - supporting_facts: List[Dict[str, Any]] (empty for TriviaQA)
    """
    split_norm = _normalize_split(split)

    # Load from HuggingFace
    try:
        raw = load_dataset("trivia_qa", setting, split=split_norm)  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"Failed to load TriviaQA from Hugging Face (setting={setting}, split={split_norm}): {e}"
        )

    # Normalize to unified schema
    ds = _normalize_triviaqa_dataset(raw)

    # Shuffle with seed if provided
    if seed is not None:
        ds = ds.shuffle(seed=seed)

    # Apply limit if provided
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    return ds
