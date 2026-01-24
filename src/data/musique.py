"""Loader for MuSiQue dataset with unified schema.

Exposes `load_musique(split, source="auto", limit=None)` returning an HF Dataset
with fields:

- id: str
- question: str
- answers: List[str]
- contexts: List[str]
- supporting_facts: List[Dict[str, Any]]

Notes:
- MuSiQue provides paragraph-level `is_supporting` flags but not sentence spans.
  We encode each supporting paragraph as `{title: <title>, sentence_id: 0}`.
"""

import json
import os
import random
from typing import Any, Dict, List, Optional

from datasets import Dataset as HFDataset  # type: ignore
from datasets import load_dataset  # type: ignore


def _normalize_split(split: str) -> str:
    s = split.lower()
    if s in {"dev", "validation"}:
        return "validation"
    return s


def _build_contexts(paragraphs: Any) -> List[str]:
    if not paragraphs:
        return []
    result: List[str] = []
    for p in paragraphs:
        if not isinstance(p, dict):
            # best-effort stringify
            result.append(str(p))
            continue
        title = str(p.get("title", "")).strip()
        text = str(p.get("paragraph_text", "")).strip()
        if title:
            result.append(f"{title}: {text}".strip())
        else:
            result.append(text)
    return result


def _build_supporting_facts(paragraphs: Any) -> List[Dict[str, Any]]:
    if not paragraphs:
        return []
    facts: List[Dict[str, Any]] = []
    for p in paragraphs:
        if isinstance(p, dict) and p.get("is_supporting"):
            facts.append({"title": str(p.get("title", "")), "sentence_id": 0})
    return facts


def _dedupe_nonempty_paragraphs(paragraphs: List[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for paragraph in paragraphs:
        text = str(paragraph).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _build_musique_rag_contexts_from_raw(examples: List[Dict[str, Any]]) -> List[str]:
    """Build a global RAG corpus from raw MuSiQue JSON examples."""
    contexts: List[str] = []
    for ex in examples:
        contexts.extend(_build_contexts(ex.get("paragraphs")))
    return contexts


def _normalize_hf_dataset(ds: HFDataset) -> HFDataset:
    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        ex_id = ex.get("id") or ex.get("_id") or ""
        # MuSiQue usually has a single string answer; still normalize to list
        answer_field = ex.get("answer")
        if isinstance(answer_field, list):
            answers = [str(a) for a in answer_field]
        elif answer_field is None:
            answers = [str(a) for a in ex.get("answers", [])]
        else:
            answers = [str(answer_field)]
        paragraphs = ex.get("paragraphs")
        return {
            "id": str(ex_id),
            "question": str(ex.get("question", "")),
            "answers": answers,
            "contexts": _build_contexts(paragraphs),
            "supporting_facts": _build_supporting_facts(paragraphs),
        }

    return ds.map(_map, remove_columns=ds.column_names, desc="normalize musique")


def load_musique(
    split: str,
    source: str = "auto",
    limit: Optional[int] = None,
    seed: Optional[int] = None,
) -> HFDataset:
    """Load MuSiQue with unified schema.

    Args:
        split: "train", "dev"/"validation", or "test" (if available).
        source: "auto" or "hf" (HF only at the moment).
        limit: optional maximum number of rows.
        seed: optional random seed for shuffling. If provided, dataset will be shuffled deterministically.
    """
    split_norm = _normalize_split(split)

    if source not in ("auto", "hf"):
        raise ValueError(f"Unsupported source for MuSiQue: {source}")

    try:
        raw = load_dataset("dgslibisey/MuSiQue", split=split_norm)  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to load MuSiQue from Hugging Face (split={split_norm}): {e}")

    ds = _normalize_hf_dataset(raw)
    # Shuffle with seed if provided
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def load_musique_rag_corpus(path: str) -> List[str]:
    """Load and build a deduplicated MuSiQue RAG corpus from a JSON/JSONL file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"MuSiQue RAG corpus file not found: {path}")

    _, ext = os.path.splitext(path)
    if ext.lower() == ".jsonl":
        contexts: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if isinstance(record, str):
                    contexts.append(record)
                    continue
                if not isinstance(record, dict):
                    contexts.append(str(record))
                    continue
                if "contents" in record:
                    contexts.append(str(record.get("contents", "")))
                    continue
                if "context" in record:
                    contexts.append(str(record.get("context", "")))
                    continue
                if "paragraphs" in record:
                    contexts.extend(_build_contexts(record.get("paragraphs")))
        return _dedupe_nonempty_paragraphs(contexts)
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            records = data.get("data") or data.get("examples") or []
        else:
            records = data

    if not isinstance(records, list):
        return []
    if records and all(isinstance(item, str) for item in records):
        return _dedupe_nonempty_paragraphs([str(item) for item in records])
    return _dedupe_nonempty_paragraphs(_build_musique_rag_contexts_from_raw(records))


def load_musique_rag_corpus_from_hf(
    split: str,
    limit: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[str]:
    """Build a deduplicated MuSiQue RAG corpus from the HF dataset."""
    ds = load_musique(split=split, source="hf", limit=limit, seed=seed)
    contexts: List[str] = []
    for ex in ds:
        contexts.extend(ex.get("contexts") or [])
    return _dedupe_nonempty_paragraphs(contexts)


def write_musique_rag_corpus_jsonl(
    path: str,
    split: str,
    limit: Optional[int] = None,
    seed: Optional[int] = None,
) -> int:
    """Write a MuSiQue RAG corpus JSONL file from the HF dataset."""
    contexts = load_musique_rag_corpus_from_hf(split=split, limit=limit, seed=seed)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for idx, ctx in enumerate(contexts):
            json.dump({"id": idx, "contents": ctx}, f, ensure_ascii=False)
            f.write("\n")
    return len(contexts)
