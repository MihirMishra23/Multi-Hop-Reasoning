"""Utilities to evaluate prediction JSONs produced by scripts/run_agent.py.

Reads a preds JSON (pandas orient="index" style) where each record contains:
  - pred: model prediction (string)
  - gold_answer: gold answer (string | list[str])
  - question: original question (string)
  - metadata: { model, split, batch_size, batch_number, type }
  - inference_params: { seed, temperature, max_tokens }

Computes answer-only metrics (EM, F1, precision, recall) and aggregates.
This module exposes pure functions so a thin CLI can wrap it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from .metrics import exact_match_score, f1_score


@dataclass
class EvalMeta:
    dataset: str
    agent: str
    llm: str
    bn: int
    bs: int
    split: str | None
    preds_path: str


def _safe_join_gold(gold: Any) -> str:
    """Convert a gold answer field into a scoring string.

    - If list-like, join with newline as requested.
    - Else, cast to string.
    """
    if gold is None:
        return ""
    if isinstance(gold, (list, tuple)):
        return "\n".join(str(x) for x in gold)
    return str(gold)


def _filename_parts_from_path(path: str) -> Tuple[str | None, str | None, str | None, int | None, int | None]:
    """Parse dataset, setting, split, bn, bs from a preds filename if present.

    Expected pattern: {dataset}_{setting}_{split}_bn={bn}_bs={bs}.json
    Returns (dataset, setting, split, bn, bs) with None on failure.
    """
    name = os.path.basename(path)
    m = re.match(r"^(?P<dataset>[^_]+)_(?P<setting>[^_]+)_(?P<split>[^_]+)_bn=(?P<bn>\d+)_bs=(?P<bs>\d+)\.json$", name)
    if not m:
        # Fallback: try without _bs=... (older files)
        m2 = re.match(r"^(?P<dataset>[^_]+)_(?P<setting>[^_]+)_(?P<split>[^_]+)_bn=(?P<bn>\d+)\.json$", name)
        if not m2:
            return None, None, None, None, None
        return (
            m2.group("dataset"),
            m2.group("setting"),
            m2.group("split"),
            int(m2.group("bn")),
            None,
        )
    return (
        m.group("dataset"),
        m.group("setting"),
        m.group("split"),
        int(m.group("bn")),
        int(m.group("bs")),
    )


def _extract_meta(preds: Dict[str, Any], preds_path: str) -> EvalMeta:
    """Extract metadata from the first record and filename as fallback."""
    # Grab first record if available
    first_record: Dict[str, Any] | None = None
    for _, rec in preds.items():
        first_record = rec
        break

    # Defaults
    dataset, setting, split, bn_from_name, bs_from_name = _filename_parts_from_path(preds_path)
    agent = None
    llm = None
    bn = None
    bs = None

    if first_record is not None:
        meta = first_record.get("metadata", {}) or {}
        agent = meta.get("type")
        llm = meta.get("model")
        # split is also inside metadata
        split = split or meta.get("split")
        bn = meta.get("batch_number")
        bs = meta.get("batch_size")

    # Path-derived agent if not present: e.g., preds/{agent}/...
    if agent is None:
        parts = os.path.normpath(preds_path).split(os.sep)
        try:
            idx = parts.index("preds")
            agent = parts[idx + 1]
        except Exception:
            agent = "unknown"

    # Prefer metadata values; fall back to filename
    if bn is None:
        bn = bn_from_name
    if bs is None:
        bs = bs_from_name

    return EvalMeta(
        dataset=dataset or "unknown",
        agent=str(agent or "unknown"),
        llm=str(llm or "unknown"),
        bn=int(bn) if isinstance(bn, int) or (isinstance(bn, str) and bn.isdigit()) else -1,
        bs=int(bs) if isinstance(bs, int) or (isinstance(bs, str) and bs.isdigit()) else -1,
        split=str(split) if split is not None else None,
        preds_path=preds_path,
    )


def evaluate_file(preds_path: str) -> Dict[str, Any]:
    """Evaluate a single preds JSON file and return metrics + metadata.

    Args:
        preds_path: Path to preds JSON saved by scripts/run_agent.py

    Returns:
        dict with keys:
          - metrics: { count, em, f1, precision, recall }
          - meta: EvalMeta as dict (dataset, agent, llm, bn, bs, split, preds_path)
    """
    with open(preds_path, "r", encoding="utf-8") as f:
        preds: Dict[str, Any] = json.load(f)

    meta = _extract_meta(preds, preds_path)

    total = 0
    sum_em = 0.0
    sum_f1 = 0.0
    sum_prec = 0.0
    sum_recall = 0.0

    for _qid, rec in preds.items():
        pred_text = rec.get("pred", "")

        # gold key variations for robustness
        gold_field = (
            rec.get("gold_answer")
            if "gold_answer" in rec
            else rec.get("answers")
            if "answers" in rec
            else rec.get("true")
        )
        gold_text = _safe_join_gold(gold_field)

        if gold_text == "":
            # If no gold present, skip this record
            continue

        em = 1.0 if exact_match_score(pred_text, gold_text) else 0.0
        f1, precision, recall = f1_score(pred_text, gold_text)

        sum_em += em
        sum_f1 += f1
        sum_prec += precision
        sum_recall += recall
        total += 1

    metrics = {
        "count": total,
        "em": (sum_em / total) if total else 0.0,
        "f1": (sum_f1 / total) if total else 0.0,
        "precision": (sum_prec / total) if total else 0.0,
        "recall": (sum_recall / total) if total else 0.0,
    }

    return {
        "metrics": metrics,
        "meta": {
            "dataset": meta.dataset,
            "agent": meta.agent,
            "llm": meta.llm,
            "bn": meta.bn,
            "bs": meta.bs,
            "split": meta.split,
            "preds_path": meta.preds_path,
        },
    }


def build_output_filename(dataset: str, agent: str, llm: str, bn: int, bs: int, timestamp: str) -> str:
    """Compose the output filename using the required schema.

    Replaces unsafe characters in llm/agent with '-'.
    """
    def safe(s: str) -> str:
        s = s.replace("/", "-")
        s = s.replace(" ", "-")
        return s

    return f"{timestamp}_{dataset}_{safe(agent)}_{safe(llm)}_bn{bn}_bs{bs}.json"


def save_results(results: Dict[str, Any], outdir: str, filename: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, filename)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return outpath
