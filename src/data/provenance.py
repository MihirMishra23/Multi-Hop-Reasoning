"""Pinned PopQA sources and run-level selection provenance."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Iterable, Optional, Union

DATASET_PROVENANCE: Dict[str, Dict[str, Any]] = {
    "popqa": {
        "hf_repo": "akariasai/PopQA",
        "revision": "098765c79ea10a2cb19c828324e33281b8336ec0",
        "file": "test.tsv",
        "upstream": "https://github.com/AlexTMallen/adaptive-retrieval",
        "license": "not specified by upstream",
    },
    "popqa_contexts": {
        "hf_repo": "ryannoonan/popqa-wikipedia-contexts",
        "revision": "1b91217d540cd4ab34e3efee1d7abe07c46f0209",
        "file": "popqa_contexts.jsonl.gz",
        "sha256": "afcc52bd4ab5ebe4f63a249fb305b21de288e8a4eafbf8c73cbd183ec3320482",
        "bytes": 45394537,
        "records": 12244,
        "upstream": "https://www.mediawiki.org/wiki/API:Main_page",
        "license": "CC BY-SA 4.0",
    },
}


def dataset_source(name: str) -> Dict[str, Any]:
    """Return a JSON-serializable copy of a pinned PopQA source record."""
    normalized = name.lower()
    if normalized not in DATASET_PROVENANCE:
        raise KeyError(f"No provenance record for dataset: {name}")
    return json.loads(json.dumps(DATASET_PROVENANCE[normalized]))


def hf_source(name: str) -> Dict[str, str]:
    source = dataset_source(name)
    return {"path": source["hf_repo"], "revision": source["revision"]}


def sha256_file(path: Union[os.PathLike, str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hf_dataset_file(name: str, filename: Optional[str] = None) -> str:
    """Resolve a pinned Hugging Face file and verify its registered checksum."""
    source = dataset_source(name)
    selected_file = filename or source.get("file")
    if not selected_file:
        raise ValueError(f"No file is registered for dataset: {name}")

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=source["hf_repo"],
        filename=selected_file,
        revision=source["revision"],
        repo_type="dataset",
    )
    expected_sha256 = source.get("sha256")
    if expected_sha256:
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Hugging Face artifact has SHA-256 {actual_sha256}, "
                f"expected {expected_sha256}: {path}"
            )
    return path


def selected_rows_provenance(
    name: str,
    ids: Iterable[Any],
    *,
    seed: Optional[int],
    setting: Optional[str],
) -> Dict[str, Any]:
    """Describe the exact ordered rows used by one evaluation run."""
    ordered_ids = [str(value) for value in ids]
    id_digest = hashlib.sha256("\n".join(ordered_ids).encode("utf-8")).hexdigest()
    return {
        "source": dataset_source(name),
        "selection": {
            "seed": seed,
            "setting": setting,
            "count": len(ordered_ids),
            "ordered_ids_sha256": id_digest,
            "ordered_ids": ordered_ids,
        },
    }
