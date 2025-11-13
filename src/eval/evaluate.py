"""Utilities to evaluate prediction JSONs produced by scripts/run_agent.py.

Reads a preds JSON with deduplicated metadata at the top level:
{
  "metadata": { model, split, batch_size, batch_number, type, retrieval },
  "inference_params": { seed, temperature, max_tokens },
  "results": {
    "qid": { pred, gold_answer, gold_evidence, question, trace, evidence }
  }
}

Computes answer-only metrics (EM, F1, precision, recall) and aggregates.
This module exposes pure functions so a thin CLI can wrap it.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

from .metrics import exact_match_score, f1_score
from src.data import get_dataset


@dataclass
class EvalMeta:
    dataset: str
    setting: str | None
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
    """Extract metadata from the top-level metadata field and filename as fallback."""
    # Defaults
    dataset, setting, split, bn_from_name, bs_from_name = _filename_parts_from_path(preds_path)
    agent = None
    llm = None
    bn = None
    bs = None

    # Extract from top-level metadata if present
    meta = preds.get("metadata", {}) or {}
    if meta:
        agent = meta.get("type")
        llm = meta.get("model")
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
        setting=str(setting) if setting is not None else None,
        agent=str(agent or "unknown"),
        llm=str(llm or "unknown"),
        bn=int(bn) if isinstance(bn, int) or (isinstance(bn, str) and bn.isdigit()) else -1,
        bs=int(bs) if isinstance(bs, int) or (isinstance(bs, str) and bs.isdigit()) else -1,
        split=str(split) if split is not None else None,
        preds_path=preds_path,
    )


def evaluate_file(
    preds_path: str,
    dataset: Optional[str] = None,
    setting: Optional[str] = None,
    split: Optional[str] = None,
    source: str = "hf",
) -> Dict[str, Any]:
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

    # Determine dataset/setting/split using overrides when provided
    dataset_name = dataset or meta.dataset
    setting_name = setting or meta.setting
    split_name = split or meta.split

    # Lazily loaded mapping from qid -> joined gold answers
    gold_by_id: Dict[str, str] | None = None

    total = 0
    sum_em = 0.0
    sum_f1 = 0.0
    sum_prec = 0.0
    sum_recall = 0.0

    # Extract results from new format (with fallback for potential edge cases)
    results = preds.get("results", preds)
    
    for _qid, rec in results.items():
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

        # If gold missing, try to load from dataset once
        if gold_text == "" and dataset_name not in (None, "unknown") and split_name is not None:
            if gold_by_id is None:
                # get_dataset requires a setting param; for datasets without setting, pass a placeholder
                effective_setting = setting_name or "na"
                try:
                    ds = get_dataset(dataset_name, effective_setting, split_name, source=source)
                    tmp: Dict[str, str] = {}
                    for row in ds:
                        ans = row.get("answers") or []
                        if isinstance(ans, list):
                            tmp[row["id"]] = "\n".join(str(x) for x in ans)
                        else:
                            tmp[row["id"]] = str(ans)
                    gold_by_id = tmp
                except Exception:
                    gold_by_id = {}
            gold_text = gold_by_id.get(str(_qid), "")

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
            "setting": meta.setting,
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
