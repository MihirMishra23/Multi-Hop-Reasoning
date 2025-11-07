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
) -> HFDataset:
    """Load MuSiQue with unified schema.

    Args:
        split: "train", "dev"/"validation", or "test" (if available).
        source: "auto" or "hf" (HF only at the moment).
        limit: optional maximum number of rows.
    """
    split_norm = _normalize_split(split)

    if source not in ("auto", "hf"):
        raise ValueError(f"Unsupported source for MuSiQue: {source}")

    try:
        raw = load_dataset("dgslibisey/MuSiQue", split=split_norm)  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"Failed to load MuSiQue from Hugging Face (split={split_norm}): {e}"
        )

    ds = _normalize_hf_dataset(raw)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


