"""Script for loading HotpotQA dataset.

This module exposes `load_hotpotqa(setting, split, source="auto", limit=None)`
that returns a Hugging Face Dataset with a unified schema:

  - id: str
  - question: str
  - answers: List[str]
  - contexts: List[str]
  - supporting_facts: List[Dict[str, Any]]

Loading preference:
  - source="auto": prefer local raw files under data/raw/hotpotqa if available,
    otherwise fall back to Hugging Face datasets ("hotpot_qa").
  - source="local": only try local raw.
  - source="hf": only try Hugging Face datasets.

No caching is performed here.
"""

import json
import os
import random
from typing import Any, Dict, List, Optional

from datasets import Dataset as HFDataset  # type: ignore
from datasets import load_dataset  # type: ignore


def _repo_root() -> str:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(this_dir)
    return os.path.abspath(os.path.join(src_dir, ".."))


def _local_hotpotqa_file(setting: str, split: str) -> Optional[str]:
    """Return path to local raw HotpotQA JSON for given setting/split if known."""
    split_norm = _normalize_split(split)
    fname: Optional[str] = None
    if setting == "distractor":
        if split_norm == "validation":
            fname = "hotpot_dev_distractor_v1.json"
        elif split_norm == "train":
            fname = "hotpot_train_v1.1.json"
        else:
            fname = None
    elif setting == "fullwiki":
        if split_norm == "validation":
            fname = "hotpot_dev_fullwiki_v1.json"
        elif split_norm == "test":
            fname = "hotpot_test_fullwiki_v1.json"
        else:
            fname = None
    if fname is None:
        return None
    path = os.path.join(_repo_root(), "data", "raw", "hotpotqa", fname)
    return path if os.path.exists(path) else None


def _normalize_split(split: str) -> str:
    if split.lower() in {"dev", "validation"}:
        return "validation"
    return split.lower()


def _build_contexts(context_field: Any) -> List[str]:
    """
    Build paragraph strings assuming the context is a dict:
      {"title": [t1, t2, ...], "sentences": [[sents1...], [sents2...], ...]}
    Titles and sentence lists are matched by index.
    """
    if not isinstance(context_field, dict):
        return []
    titles = context_field.get("title")
    sentences = context_field.get("sentences")
    if not isinstance(titles, list) or not isinstance(sentences, list):
        return []
    contexts: List[str] = []
    for i, title in enumerate(titles):
        sents_i = sentences[i] if i < len(sentences) else []
        sent_list = [s for s in (sents_i or [])]
        paragraph = f"{title}: " + " ".join(sent_list).strip()
        contexts.append(paragraph.strip())
    return contexts


def _build_supporting_facts(sf_field: Any) -> List[Dict[str, Any]]:
    if not sf_field:
        return []
    titles_and_ids = zip(sf_field['title'], sf_field['sent_id'])
    result: List[Dict[str, Any]] = []
    for item in titles_and_ids:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        title, sent_id = item
        result.append({"title": str(title), "sentence_id": int(sent_id)})
    return result


def _normalize_examples_pylist(examples: List[Dict[str, Any]]) -> HFDataset:
    rows: List[Dict[str, Any]] = []
    for ex in examples:
        ex_id = ex.get("_id") or ex.get("id") or ""
        question = ex.get("question") or ""
        answer = ex.get("answer")
        answers: List[str]
        if isinstance(answer, str):
            answers = [answer]
        elif isinstance(answer, list):
            answers = [str(a) for a in answer]
        else:
            answers = ex.get("answers") or []
            answers = [str(a) for a in answers]
        contexts = _build_contexts(ex.get("context"))
        supporting_facts = _build_supporting_facts(ex.get("supporting_facts"))
        rows.append(
            {
                "id": str(ex_id),
                "question": str(question),
                "answers": answers,
                "contexts": contexts,
                "supporting_facts": supporting_facts,
            }
        )
    return HFDataset.from_list(rows)


def _normalize_hf_dataset(ds: HFDataset) -> HFDataset:
    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        ex_id = ex.get("_id") or ex.get("id") or ""
        answer = ex.get("answer")
        if isinstance(answer, str):
            answers = [answer]
        elif isinstance(answer, list):
            answers = [str(a) for a in answer]
        else:
            answers = [str(a) for a in ex.get("answers", [])]
        return {
            "id": str(ex_id),
            "question": str(ex.get("question", "")),
            "answers": answers,
            "contexts": _build_contexts(ex.get("context")),
            "supporting_facts": _build_supporting_facts(ex.get("supporting_facts")),
        }

    return ds.map(_map, remove_columns=ds.column_names, desc="normalize hotpotqa")


def load_hotpotqa(
    setting: str,
    split: str,
    source: str = "auto",
    limit: Optional[int] = None,
    seed: Optional[int] = None,
) -> HFDataset:
    """Load HotpotQA with unified schema.

    Args:
        setting: "distractor" or "fullwiki".
        split: "train", "dev"/"validation", or "test" (where available).
        source: "auto" (prefer local), "local", or "hf".
        limit: optional max number of rows to return.
        seed: optional random seed for shuffling. If provided, dataset will be shuffled deterministically.
    """

    split_norm = _normalize_split(split)

    # Try local raw
    if source in ("auto", "local"):
        local_path = _local_hotpotqa_file(setting, split_norm)
        if local_path is not None:
            with open(local_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                ds = _normalize_examples_pylist(data)
                # Shuffle with seed if provided
                if seed is not None:
                    ds = ds.shuffle(seed=seed)
                if limit is not None:
                    ds = ds.select(range(min(limit, len(ds))))
                return ds
        if source == "local":
            raise FileNotFoundError(
                f"Local HotpotQA file not found for setting={setting} split={split_norm}"
            )

    # Fallback to Hugging Face
    hf_split = split_norm
    try:
        raw = load_dataset("hotpot_qa", setting, split=hf_split)  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"Failed to load HotpotQA from Hugging Face (setting={setting}, split={hf_split}): {e}"
        )
    ds = _normalize_hf_dataset(raw)
    # Shuffle with seed if provided
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds