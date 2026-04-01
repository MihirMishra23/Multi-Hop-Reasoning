"""Loader for PopQA dataset with unified schema.

Exposes `load_popqa(split, source="auto", limit=None)` returning an HF Dataset
with fields:

- id: str
- question: str
- answers: List[str]
- contexts: List[str]
- context_titles: List[str]  (parallel to contexts, for splitting logic)
- supporting_facts: List[Dict[str, Any]]  (empty for PopQA)
- golden_contexts: List[str]

Notes:
- PopQA does not provide supporting facts, so supporting_facts is always empty.
- Contexts are loaded from a local Wikipedia corpus JSON file.
- context_titles stores the titles separately to avoid fragile string parsing during splitting.
- Context splitting (sentence grouping >= 800 chars) is handled by eval_multihop.py, not here.
"""

import json
import os
from typing import Any, Dict, List, Optional

from datasets import Dataset as HFDataset  # type: ignore
from datasets import load_dataset  # type: ignore

# Path to the PopQA Wikipedia corpus
POPQA_CORPUS_PATH = "/home/rtn27/Multi-Hop-Reasoning/wiki_dump_popqa_s_42_1k.json"


def _normalize_split(split: str) -> str:
    """Normalize split names (dev -> validation)."""
    s = split.lower()
    if s in {"dev", "validation"}:
        return "test"
    return s


def _load_popqa_corpus(corpus_path: str = POPQA_CORPUS_PATH) -> List[Dict[str, Any]]:
    """Load the PopQA Wikipedia corpus from JSON file.

    Args:
        corpus_path: Path to the corpus JSON file

    Returns:
        List of corpus entries with 'title' and 'paragraphs' fields
    """
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"PopQA corpus not found: {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    return corpus


def _build_context_titles_from_corpus(corpus: List[Dict[str, Any]]) -> List[str]:
    """Extract title list from corpus.

    Args:
        corpus: List of corpus entries with 'title' and 'paragraphs'

    Returns:
        List of titles (parallel to contexts returned by _build_contexts_from_corpus)
    """
    titles: List[str] = []
    for entry in corpus:
        if isinstance(entry, dict):
            title = entry.get("title", "")
            titles.append(str(title))
    return titles


def _build_contexts_from_corpus(corpus: List[Dict[str, Any]]) -> List[str]:
    """Build context strings from corpus.

    Args:
        corpus: List of corpus entries with 'title' and 'paragraphs'

    Returns:
        List of concatenated paragraph strings (one per corpus entry)
    """
    contexts: List[str] = []
    for entry in corpus:
        if not isinstance(entry, dict):
            continue

        paragraphs = entry.get("paragraphs", [])
        if not isinstance(paragraphs, list):
            continue

        # Concatenate all paragraphs with double newline separator
        text = "\n\n".join(str(p).strip() for p in paragraphs if str(p).strip())
        if text:
            contexts.append(text)

    return contexts


def _build_answers(answer_field: Any) -> List[str]:
    """Build answer list from PopQA answer field.

    Args:
        answer_field: Answer field from dataset (can be string, JSON string, or list)

    Returns:
        List of answer strings
    """
    if isinstance(answer_field, list):
        # If it's already a list, just clean up the strings
        result = []
        for a in answer_field:
            # Handle case where list item might be a JSON string
            if isinstance(a, str) and a.strip().startswith('['):
                try:
                    parsed = json.loads(a)
                    if isinstance(parsed, list):
                        result.extend([str(x).strip() for x in parsed if str(x).strip()])
                    else:
                        result.append(str(a).strip())
                except (json.JSONDecodeError, ValueError):
                    result.append(str(a).strip())
            else:
                s = str(a).strip()
                if s:
                    result.append(s)
        return result
    elif answer_field is not None:
        # Try to parse as JSON if it's a string representation of a list
        answer_str = str(answer_field).strip()
        if answer_str.startswith('['):
            try:
                parsed = json.loads(answer_str)
                if isinstance(parsed, list):
                    return [str(a).strip() for a in parsed if str(a).strip()]
            except (json.JSONDecodeError, ValueError):
                pass
        return [answer_str]
    return []


def _normalize_hf_dataset(ds: HFDataset, contexts: List[str], context_titles: List[str]) -> HFDataset:
    """Normalize PopQA HF dataset to unified schema.

    Args:
        ds: Raw HuggingFace dataset
        contexts: Pre-loaded contexts from corpus (aligned by index)
        context_titles: Pre-loaded context titles from corpus (aligned by index)
    """
    def _map(ex: Dict[str, Any], idx: int) -> Dict[str, Any]:
        ex_id = ex.get("id") or ex.get("question_id") or ""
        question = ex.get("question", "")

        # Build answers from the answer field
        answer_field = ex.get("possible_answers")
        answers = _build_answers(answer_field)

        # Get the context for this specific example (corpus is aligned by index)
        ex_contexts = [contexts[idx]] if idx < len(contexts) else []
        ex_context_titles = [context_titles[idx]] if idx < len(context_titles) else []

        return {
            "id": str(ex_id),
            "question": str(question),
            "answers": answers,
            "contexts": ex_contexts,
            "context_titles": ex_context_titles,  # Parallel list of titles for splitting logic
            "golden_contexts": ex_contexts,  # No ground truth supporting docs, so use all
            "supporting_facts": [],  # PopQA has no sentence-level supporting facts
        }

    # Map to unified schema with indices, avoid cache conflicts
    return ds.map(
        _map,
        with_indices=True,
        remove_columns=ds.column_names,
        desc="normalize popqa",
        load_from_cache_file=False,
        keep_in_memory=True,
    )


def load_popqa(
    split: str,
    source: str = "auto",
    limit: Optional[int] = None,
    seed: Optional[int] = None,
    setting: Optional[str] = None,
    corpus_path: str = POPQA_CORPUS_PATH,
) -> HFDataset:
    """Load PopQA with unified schema.

    Args:
        split: "train", "dev"/"validation", or "test"
        source: "auto" or "hf" (only "hf" supported for PopQA)
        limit: optional max number of rows to return
        seed: optional random seed for shuffling
        setting: dataset setting (unused for PopQA)
        corpus_path: path to the PopQA Wikipedia corpus JSON

    Returns:
        HFDataset with unified schema
    """
    split_norm = _normalize_split(split)

    # Load the Wikipedia corpus
    corpus = _load_popqa_corpus(corpus_path)
    contexts = _build_contexts_from_corpus(corpus)
    context_titles = _build_context_titles_from_corpus(corpus)

    # Load from Hugging Face
    # PopQA is available on HF as "akariasai/PopQA"
    try:
        raw = load_dataset("akariasai/PopQA", split=split_norm)  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"Failed to load PopQA from Hugging Face (split={split_norm}): {e}"
        )

    # Shuffle with seed if provided (before normalizing to avoid duplicating heavy context data)
    if seed is not None:
        raw = raw.shuffle(seed=seed)

    # Limit results if requested (before normalizing)
    if limit is not None:
        raw = raw.select(range(min(limit, len(raw))))

    # Limit contexts and context_titles to match the dataset length
    # (corpus was pre-built with same seed/limit, so indices align)
    dataset_len = len(raw)
    contexts = contexts[:dataset_len]
    context_titles = context_titles[:dataset_len]

    # Normalize to unified schema (adds contexts to each example)
    ds = _normalize_hf_dataset(raw, contexts, context_titles)

    return ds
