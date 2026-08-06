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

import gzip
import json
import os
import warnings
from typing import Any, Dict, List, Optional

from datasets import Dataset as HFDataset  # type: ignore
from datasets import load_dataset  # type: ignore

# Path to the PopQA Wikipedia corpus
POPQA_CORPUS_PATH = "/share/j_sun/lmlm_multihop/popqa/popqa_corpus_1000_ex_seed_42.json"


def _resolve_popqa_corpus_path(corpus_path: Optional[str]) -> str:
    """Resolve an explicit path, the environment override, or the legacy artifact."""
    return corpus_path or os.environ.get("POPQA_CORPUS_PATH") or POPQA_CORPUS_PATH


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

    opener = gzip.open if corpus_path.endswith(".gz") else open
    with opener(corpus_path, "rt", encoding="utf-8") as stream:
        if corpus_path.endswith((".jsonl", ".jsonl.gz")):
            corpus = [json.loads(line) for line in stream if line.strip()]
        else:
            payload = json.load(stream)
            if isinstance(payload, list):
                corpus = payload
            elif isinstance(payload, dict):
                corpus = payload.get("articles") or payload.get("records") or []
            else:
                corpus = []

    if not isinstance(corpus, list):
        raise ValueError(f"PopQA corpus must contain a list of records: {corpus_path}")
    return corpus


def _build_corpus_index(corpus: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """Index usable articles by requested and canonical title.

    The historical artifact was deduplicated and saved in an order unrelated to
    the shuffled PopQA rows.  A title-keyed join is therefore required even for
    reproducing the original 1,000-row evaluation subset.
    """
    index: Dict[str, Dict[str, str]] = {}
    for entry in corpus:
        if not isinstance(entry, dict) or entry.get("status", "ok") != "ok":
            continue
        paragraphs = entry.get("paragraphs", [])
        if not isinstance(paragraphs, list):
            continue
        text = "\n\n".join(
            str(paragraph).strip() for paragraph in paragraphs if str(paragraph).strip()
        )
        if not text:
            continue
        requested_title = str(entry.get("requested_title") or entry.get("title") or "").strip()
        canonical_title = str(entry.get("title") or requested_title).strip()
        article = {"title": canonical_title, "text": text}
        for title in (requested_title, canonical_title):
            if title:
                index.setdefault(title, article)
    return index


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
            if isinstance(a, str) and a.strip().startswith("["):
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
        if answer_str.startswith("["):
            try:
                parsed = json.loads(answer_str)
                if isinstance(parsed, list):
                    return [str(a).strip() for a in parsed if str(a).strip()]
            except (json.JSONDecodeError, ValueError):
                pass
        return [answer_str]
    return []


def _normalize_hf_dataset(ds: HFDataset, corpus_index: Dict[str, Dict[str, str]]) -> HFDataset:
    """Normalize PopQA HF dataset to unified schema.

    Args:
        ds: Raw HuggingFace dataset
        corpus_index: Wikipedia articles keyed by PopQA's subject title
    """

    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        ex_id = ex.get("id") or ex.get("question_id") or ""
        question = ex.get("question", "")

        # Build answers from the answer field
        answer_field = ex.get("possible_answers")
        answers = _build_answers(answer_field)

        requested_title = str(ex.get("s_wiki_title") or ex.get("subj") or "").strip()
        article = corpus_index.get(requested_title)
        ex_contexts = [article["text"]] if article else []
        ex_context_titles = [article["title"]] if article else []

        return {
            "id": str(ex_id),
            "question": str(question),
            "answers": answers,
            "contexts": ex_contexts,
            "context_titles": ex_context_titles,  # Parallel list of titles for splitting logic
            "golden_contexts": ex_contexts,  # No ground truth supporting docs, so use all
            "supporting_facts": [],  # PopQA has no sentence-level supporting facts
        }

    # Avoid cache conflicts with older positional PopQA normalization.
    return ds.map(
        _map,
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
    corpus_path: Optional[str] = None,
) -> HFDataset:
    """Load PopQA with unified schema.

    Args:
        split: "train", "dev"/"validation", or "test"
        source: "auto" or "hf" (only "hf" supported for PopQA)
        limit: optional max number of rows to return
        seed: optional random seed for shuffling
        setting: dataset setting (unused for PopQA)
        corpus_path: path to a PopQA Wikipedia corpus JSON/JSONL file. If omitted,
            POPQA_CORPUS_PATH from the environment is used before the bundled
            legacy artifact.

    Returns:
        HFDataset with unified schema
    """
    split_norm = _normalize_split(split)

    # Load the Wikipedia corpus
    resolved_corpus_path = _resolve_popqa_corpus_path(corpus_path)
    corpus = _load_popqa_corpus(resolved_corpus_path)
    corpus_index = _build_corpus_index(corpus)

    # Load from Hugging Face
    # PopQA is available on HF as "akariasai/PopQA"
    try:
        raw = load_dataset("akariasai/PopQA", split=split_norm)  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to load PopQA from Hugging Face (split={split_norm}): {e}")

    # Shuffle with seed if provided (before normalizing to avoid duplicating heavy context data)
    if seed is not None:
        raw = raw.shuffle(seed=seed)

    # Limit results if requested (before normalizing)
    if limit is not None:
        raw = raw.select(range(min(limit, len(raw))))

    requested_titles = {str(title).strip() for title in raw["s_wiki_title"] if str(title).strip()}
    missing_titles = requested_titles.difference(corpus_index)
    if missing_titles:
        preview = ", ".join(sorted(missing_titles)[:3])
        warnings.warn(
            f"PopQA corpus covers {len(requested_titles) - len(missing_titles)}/"
            f"{len(requested_titles)} requested Wikipedia titles; missing examples: {preview}",
            RuntimeWarning,
        )

    # Normalize to unified schema (adds contexts to each example)
    ds = _normalize_hf_dataset(raw, corpus_index)

    return ds
