"""Durable, tail-able progress artifacts for long-running evaluations."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, MutableSet, Optional

from .metrics import exact_match_score, f1_score


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic_write_json(path: str, payload: Dict[str, Any], *, indent: int = 2) -> None:
    """Replace ``path`` atomically so readers never observe half-written JSON."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temporary_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(temporary_path, "w", encoding="utf-8") as destination:
            json.dump(payload, destination, ensure_ascii=False, indent=indent)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _gold_answers(result: Dict[str, Any]) -> List[str]:
    gold = result.get("gold_answer")
    if gold is None:
        return []
    if isinstance(gold, (list, tuple)):
        return [str(answer) for answer in gold]
    return [str(gold)]


def score_result(result: Dict[str, Any], method: str) -> Dict[str, Any]:
    """Score one result with the run's approximate online answer metric."""
    if method == "ircot":
        # This is the exact answer scorer used by the faithful IRCoT port.
        from .ircot_official_metrics import evaluate_predictions

        metrics = evaluate_predictions([result])
        return {
            "metric": "official_ircot_drop",
            "em": float(metrics["em"]),
            "f1": float(metrics["f1"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
        }

    prediction = str(result.get("pred") or "")
    answers = _gold_answers(result)
    best = {"em": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    for answer in answers:
        em = float(exact_match_score(prediction, answer))
        f1, precision, recall = f1_score(prediction, answer)
        if em > best["em"]:
            best["em"] = em
        if f1 > best["f1"]:
            best.update(f1=float(f1), precision=float(precision), recall=float(recall))
    return {"metric": "answer_em_f1", **best}


def summarize_scores(scores: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(scores)
    count = len(rows)
    metric_names = {str(row.get("metric") or "unknown") for row in rows}
    return {
        "metric": metric_names.pop() if len(metric_names) == 1 else "mixed",
        "count": count,
        "em": sum(float(row.get("em") or 0.0) for row in rows) / count if count else 0.0,
        "f1": sum(float(row.get("f1") or 0.0) for row in rows) / count if count else 0.0,
        "precision": (
            sum(float(row.get("precision") or 0.0) for row in rows) / count if count else 0.0
        ),
        "recall": (
            sum(float(row.get("recall") or 0.0) for row in rows) / count if count else 0.0
        ),
    }


def score_results(results: Dict[str, Dict[str, Any]], method: str) -> List[Dict[str, Any]]:
    return [score_result(result, method) for result in results.values()]


def completion_record(
    qid: str,
    result: Dict[str, Any],
    method: str,
    score: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    trace = result.get("trace") or []
    last_step = trace[-1] if trace and isinstance(trace[-1], dict) else {}
    raw_completion = (
        last_step.get("raw_response")
        or last_step.get("answer")
        or result.get("pred")
        or ""
    )
    return {
        "timestamp_utc": utc_now(),
        "qid": str(qid),
        "pred": result.get("pred") or "",
        "gold_answer": result.get("gold_answer"),
        "completion": raw_completion,
        "trace_steps": len(trace),
        **(score or score_result(result, method)),
    }


def load_completion_ids(path: str) -> MutableSet[str]:
    ids: MutableSet[str] = set()
    if not os.path.isfile(path):
        return ids
    with open(path, "r", encoding="utf-8") as source:
        for line in source:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("qid") is not None:
                ids.add(str(record["qid"]))
    return ids


def append_completion_records(
    path: str,
    batch_results: Dict[str, Dict[str, Any]],
    method: str,
    logged_ids: MutableSet[str],
) -> List[Dict[str, Any]]:
    """Append new full completions to JSONL and return their score records."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    records: List[Dict[str, Any]] = []
    with open(path, "a", encoding="utf-8") as destination:
        for qid, result in batch_results.items():
            if str(qid) in logged_ids:
                continue
            record = completion_record(str(qid), result, method)
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            logged_ids.add(str(qid))
        destination.flush()
        os.fsync(destination.fileno())
    return records


def write_progress(
    path: str,
    *,
    completed: int,
    expected: int,
    scores: Iterable[Dict[str, Any]],
    checkpoint_path: str,
    completions_path: str,
    status: str,
    last_qids: Optional[List[str]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    rolling = summarize_scores(scores)
    payload: Dict[str, Any] = {
        "updated_at_utc": utc_now(),
        "status": status,
        "completed": completed,
        "expected": expected,
        "fraction": completed / expected if expected else 1.0,
        "rolling": rolling,
        "checkpoint_path": checkpoint_path,
        "completions_path": completions_path,
        "last_qids": last_qids or [],
    }
    if error:
        payload["error"] = error
    atomic_write_json(path, payload)
    return payload


__all__ = [
    "append_completion_records",
    "atomic_write_json",
    "completion_record",
    "load_completion_ids",
    "score_result",
    "score_results",
    "summarize_scores",
    "utc_now",
    "write_progress",
]
