"""Loader for the exact processed evaluation files released with IRCoT.

The upstream files are JSONL records with ``question_id``, ``question_text``,
``answers_objects``, and normalized paragraph ``contexts``.  Their row order is
part of the released evaluation protocol and is intentionally not shuffled.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from datasets import Dataset as HFDataset  # type: ignore


def _answer_strings(answer_objects: Any) -> List[str]:
    answers: List[str] = []
    for answer_object in answer_objects or []:
        if not isinstance(answer_object, dict):
            continue
        number = str(answer_object.get("number", "")).strip()
        if number:
            answers.append(number)
            continue
        spans = answer_object.get("spans") or []
        if spans:
            answers.extend(str(span) for span in spans)
            continue
        date = answer_object.get("date") or {}
        if isinstance(date, dict) and any(date.get(key) for key in ("day", "month", "year")):
            answers.append(
                "-".join(str(date.get(key, "")) for key in ("day", "month", "year"))
            )
    return answers or [""]


def _normalize_contexts(contexts: Any) -> Dict[str, List[Any]]:
    formatted: List[str] = []
    golden: List[str] = []
    supporting_facts: List[Dict[str, Any]] = []
    for paragraph in contexts or []:
        if not isinstance(paragraph, dict):
            continue
        title = str(paragraph.get("title", "")).strip()
        text = str(paragraph.get("paragraph_text", "")).strip()
        rendered = f"{title}: {text}".strip() if title else text
        formatted.append(rendered)
        if paragraph.get("is_supporting"):
            golden.append(rendered)
            supporting_facts.append({"title": title, "sentence_id": 0})
    return {
        "contexts": formatted,
        "golden_contexts": golden,
        "supporting_facts": supporting_facts,
    }


def load_ircot_official_evaluation(path: str) -> HFDataset:
    """Load a released ``{dev|test}_subsampled.jsonl`` without resampling."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            qid = str(record.get("question_id", ""))
            question = str(record.get("question_text", ""))
            if not qid or not question:
                raise ValueError(f"Invalid IRCoT evaluation row {line_number} in {path}")
            normalized_contexts = _normalize_contexts(record.get("contexts"))
            rows.append(
                {
                    "id": qid,
                    "question": question,
                    "answers": _answer_strings(record.get("answers_objects")),
                    "answer_objects": record.get("answers_objects") or [],
                    **normalized_contexts,
                }
            )
    if not rows:
        raise ValueError(f"IRCoT evaluation file is empty: {path}")
    return HFDataset.from_list(rows)


__all__ = ["load_ircot_official_evaluation"]
