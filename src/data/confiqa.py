"""Portable ConFiQA-MR loader with deterministic counterfactual subsets."""

import ast
import hashlib
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset as HFDataset  # type: ignore

from .provenance import dataset_source, download_verified_file, sha256_file

logger = logging.getLogger(__name__)
CONFLICT_FREE_SETTINGS = {"cf_100_conflict_free", "cf_356_conflict_free"}
VALID_SETTINGS = {"orig", "cf", "cf_100", "cf_500", *CONFLICT_FREE_SETTINGS}
_SETTING_ALIASES = {
    "CF-100-conflict-free": "cf_100_conflict_free",
    "CF-356-conflict-free": "cf_356_conflict_free",
}
_CONDITION_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "artifacts"
    / "confiqa_conflict_free_conditions.json"
)


def canonical_confiqa_setting(setting: str) -> str:
    return _SETTING_ALIASES.get(setting, setting)


@lru_cache(maxsize=1)
def load_conflict_free_manifest() -> Dict[str, Any]:
    with _CONDITION_MANIFEST.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def conflict_free_condition_metadata(setting: str) -> Optional[Dict[str, Any]]:
    setting = canonical_confiqa_setting(setting)
    if setting not in CONFLICT_FREE_SETTINGS:
        return None
    manifest = load_conflict_free_manifest()
    return {
        "manifest_path": str(_CONDITION_MANIFEST.relative_to(Path(__file__).resolve().parents[2])),
        "manifest_sha256": sha256_file(_CONDITION_MANIFEST),
        "schema_version": manifest["schema_version"],
        "selection_universe": manifest["selection_universe"],
        "smoke_query_selection": manifest["smoke_query_selection"],
        "selection_algorithm": manifest["selection_algorithm"],
        "condition": manifest["conditions"][setting],
    }


def _parse_triplets(triplet_str: str) -> List[Tuple[str, str, str]]:
    if not triplet_str:
        return []
    try:
        parsed = ast.literal_eval(triplet_str)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    triplets: List[Tuple[str, str, str]] = []
    for item in parsed:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            triplets.append((str(item[0]), str(item[1]), str(item[2])))
    return triplets


def _resolve_confiqa_path(
    source: str, confiqa_path: Optional[str], cache_dir: Optional[str]
) -> str:
    """Resolve an explicit local override or the pinned canonical download."""
    source_norm = source.lower()
    if source_norm not in {"auto", "local", "hf", "remote"}:
        raise ValueError(f"Unsupported ConFiQA source: {source}")

    explicit_path = confiqa_path or os.environ.get("CONFIQA_PATH")
    if explicit_path:
        expanded = os.path.abspath(os.path.expanduser(explicit_path))
        if not os.path.exists(expanded):
            raise FileNotFoundError(f"ConFiQA dataset not found: {expanded}")
        expected = dataset_source("confiqa")["sha256"]
        actual = sha256_file(expanded)
        if actual != expected:
            raise ValueError(
                f"ConFiQA file has SHA-256 {actual}, expected canonical {expected}: {expanded}"
            )
        return expanded

    if source_norm == "local":
        raise FileNotFoundError(
            "source='local' requires confiqa_path=... or the CONFIQA_PATH environment variable"
        )

    provenance = dataset_source("confiqa")
    return download_verified_file(
        url=provenance["url"],
        sha256=provenance["sha256"],
        relative_path="confiqa/ConFiQA-MR.json",
        cache_dir=cache_dir,
    )


def _load_confiqa(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of ConFiQA examples in {path}")
    return data


def _counterfactual_cutoff(setting: str) -> Optional[int]:
    setting = canonical_confiqa_setting(setting)
    if setting == "orig":
        return 0
    if setting == "cf":
        return None
    if setting == "cf_100":
        return 100
    if setting == "cf_500":
        return 500
    if setting in CONFLICT_FREE_SETTINGS:
        return None
    raise ValueError(
        f"Unsupported ConFiQA setting {setting!r}; choose from {sorted(VALID_SETTINGS)}"
    )


def _ordered_source_indices(size: int, seed: Optional[int]) -> List[int]:
    """Match Hugging Face Dataset.shuffle while preserving raw source IDs."""
    indices = HFDataset.from_dict({"source_index": list(range(size))})
    if seed is not None:
        indices = indices.shuffle(seed=seed)
    return [int(value) for value in indices["source_index"]]


def _normalize_confiqa(
    data: List[Dict[str, Any]], source_indices: List[int], setting: str
) -> HFDataset:
    setting = canonical_confiqa_setting(setting)
    cutoff = _counterfactual_cutoff(setting)
    selected_cf_ids = None
    condition_label = None
    if setting in CONFLICT_FREE_SETTINGS:
        condition = load_conflict_free_manifest()["conditions"][setting]
        selected_cf_ids = set(int(value) for value in condition["selected_cf_source_ids"])
        condition_label = condition["label"]
    examples: List[Dict[str, Any]] = []

    for eval_position, source_index in enumerate(source_indices):
        ex = data[source_index]
        use_cf = (
            setting == "cf"
            or (selected_cf_ids is not None and source_index in selected_cf_ids)
            or (selected_cf_ids is None and cutoff is not None and eval_position < cutoff)
        )
        prefix = "cf" if use_cf else "orig"
        context = str(ex.get(f"{prefix}_context", ""))
        answer = str(ex.get(f"{prefix}_answer", ""))
        aliases = ex.get(f"{prefix}_alias", [])
        triplets = _parse_triplets(str(ex.get(f"{prefix}_path_labeled", "")))

        answers = [answer] if answer else []
        if isinstance(aliases, list):
            answers.extend(str(alias).strip() for alias in aliases if str(alias).strip())
        unique_answers = list(dict.fromkeys(answers))

        examples.append(
            {
                "id": str(source_index),
                "source_index": source_index,
                "eval_position": eval_position,
                "confiqa_setting": setting,
                "confiqa_condition_label": condition_label,
                "is_counterfactual": use_cf,
                "question": str(ex.get("question", "")),
                "answers": unique_answers,
                "contexts": [context] if context else [],
                "context_titles": [f"ConFiQA {source_index}"],
                "golden_contexts": [context] if context else [],
                "supporting_facts": [],
                "golden_triplets": triplets,
            }
        )
    return HFDataset.from_list(examples)


def load_confiqa(
    split: str = "test",
    source: str = "auto",
    limit: Optional[int] = None,
    seed: Optional[int] = None,
    setting: str = "orig",
    confiqa_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> HFDataset:
    """Load ConFiQA-MR from its pinned public source.

    The dataset is ordered first, then the counterfactual setting is applied.
    Consequently ``orig``, ``cf_100`` and ``cf_500`` always contain the same
    questions in the same order, and a 1,000-row evaluation has exactly 0, 100
    and 500 counterfactual rows respectively.
    """
    setting = canonical_confiqa_setting(setting)
    if split.lower() not in {"test", "dev", "validation"}:
        raise ValueError("ConFiQA-MR has one evaluation split; use test/dev/validation")
    if setting not in VALID_SETTINGS:
        _counterfactual_cutoff(setting)

    path = _resolve_confiqa_path(source, confiqa_path, cache_dir)
    raw_data = _load_confiqa(path)
    source_indices = _ordered_source_indices(len(raw_data), seed)
    if setting in CONFLICT_FREE_SETTINGS:
        manifest_universe = load_conflict_free_manifest()["selection_universe"]
        if seed != manifest_universe["seed"]:
            raise ValueError(
                f"{setting} is pinned to seed {manifest_universe['seed']}, received {seed}"
            )
        retained_count = int(manifest_universe["retained_count"])
        retained_ids = source_indices[:retained_count]
        retained_hash = hashlib.sha256(
            "\n".join(str(value) for value in retained_ids).encode("utf-8")
        ).hexdigest()
        if retained_ids != manifest_universe["ordered_source_ids"]:
            raise ValueError(
                "Pinned ConFiQA order does not match the conflict-free condition manifest "
                f"(actual SHA-256 {retained_hash})"
            )
    if limit is not None:
        source_indices = source_indices[: min(limit, len(source_indices))]
    dataset = _normalize_confiqa(raw_data, source_indices, setting)
    logger.info(
        "Loaded %d ConFiQA rows (setting=%s, seed=%s, counterfactual=%d)",
        len(dataset),
        setting,
        seed,
        sum(bool(value) for value in dataset["is_counterfactual"]),
    )
    return dataset
