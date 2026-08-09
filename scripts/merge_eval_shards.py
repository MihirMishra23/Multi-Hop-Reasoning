#!/usr/bin/env python3
"""Merge disjoint evaluator shards with strict coverage/provenance checks."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


START_INDEX_PATTERN = re.compile(r"_i(\d+)(?:_|\.)")
INVARIANT_METADATA_KEYS = (
    "dataset",
    "setting",
    "dataset_source",
    "split",
    "type",
    "model",
    "model-path",
    "resolved_model_id",
    "model_revision",
    "code_commit",
    "retrieval",
    "ircot",
)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
        raise ValueError(f"Shard does not contain a results object: {path}")
    return payload


def _start_index(path: str) -> int:
    match = START_INDEX_PATTERN.search(os.path.basename(path))
    if not match:
        raise ValueError(f"Cannot parse _i<start-index> from shard filename: {path}")
    return int(match.group(1))


def _ordered_ids(payload: Dict[str, Any]) -> List[str]:
    return [str(value) for value in payload["results"].keys()]


def _validate_invariants(reference: Dict[str, Any], candidate: Dict[str, Any], path: str) -> None:
    reference_metadata = reference.get("metadata") or {}
    candidate_metadata = candidate.get("metadata") or {}
    for key in INVARIANT_METADATA_KEYS:
        if candidate_metadata.get(key) != reference_metadata.get(key):
            raise ValueError(f"Shard metadata mismatch for {key!r}: {path}")
    if candidate.get("inference_params") != reference.get("inference_params"):
        raise ValueError(f"Shard inference parameters do not match: {path}")


def _atomic_write(path: str, payload: Dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)
        output.write("\n")
    os.replace(temporary, destination)


def merge_shards(
    paths: Iterable[str],
    *,
    expected_count: int,
    expected_shards: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    ordered_paths = sorted(set(paths), key=_start_index)
    if len(ordered_paths) != expected_shards:
        raise ValueError(f"Expected {expected_shards} shards, found {len(ordered_paths)}")

    reference = _read_json(ordered_paths[0])
    merged_results: Dict[str, Any] = {}
    shard_records: List[Dict[str, Any]] = []
    cursor = 0

    for path in ordered_paths:
        payload = _read_json(path)
        _validate_invariants(reference, payload, path)
        metadata = payload.get("metadata") or {}
        start_index = _start_index(path)
        expected_examples = int(metadata.get("expected_examples") or len(payload["results"]))
        if start_index != cursor:
            raise ValueError(
                f"Shard coverage is not contiguous: expected start {cursor}, found {start_index} in {path}"
            )
        if len(payload["results"]) != expected_examples:
            raise ValueError(
                f"Shard is incomplete: {path} has {len(payload['results'])}/{expected_examples} results"
            )

        ids = _ordered_ids(payload)
        provenance_ids = (
            ((metadata.get("dataset_provenance") or {}).get("selection") or {}).get("ordered_ids")
            or []
        )
        if provenance_ids and [str(value) for value in provenance_ids] != ids:
            raise ValueError(f"Shard result order disagrees with dataset provenance: {path}")
        duplicates = set(ids).intersection(merged_results)
        if duplicates:
            raise ValueError(f"Duplicate question IDs across shards: {sorted(duplicates)[:3]}")
        merged_results.update(payload["results"])
        shard_records.append(
            {
                "path": os.path.abspath(path),
                "start_index": start_index,
                "expected_examples": expected_examples,
                "results": len(payload["results"]),
                "ordered_ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
            }
        )
        cursor += expected_examples

    if cursor != expected_count or len(merged_results) != expected_count:
        raise ValueError(
            f"Merged coverage is incomplete: covered {cursor}, unique results {len(merged_results)}, "
            f"expected {expected_count}"
        )

    merged = deepcopy(reference)
    merged["results"] = merged_results
    metadata = merged.setdefault("metadata", {})
    ordered_ids = list(merged_results)
    source_provenance = (metadata.get("dataset_provenance") or {}).get("source")
    selection = ((metadata.get("dataset_provenance") or {}).get("selection") or {})
    metadata["total_examples"] = expected_count
    metadata["expected_examples"] = expected_count
    metadata["generated_at_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    metadata["dataset_provenance"] = {
        "source": source_provenance,
        "selection": {
            "seed": selection.get("seed"),
            "setting": selection.get("setting"),
            "count": expected_count,
            "ordered_ids_sha256": hashlib.sha256("\n".join(ordered_ids).encode("utf-8")).hexdigest(),
            "ordered_ids": ordered_ids,
        },
    }
    metadata["merge"] = {
        "implementation": "strict_contiguous_eval_shards_v1",
        "shard_count": len(shard_records),
        "shards": shard_records,
    }
    return merged, shard_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--results-output")
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    args = parser.parse_args()

    paths = glob.glob(args.input_glob)
    merged, shard_records = merge_shards(
        paths,
        expected_count=args.expected_count,
        expected_shards=args.expected_shards,
    )
    _atomic_write(args.output, merged)

    if args.results_output:
        method = (merged.get("metadata") or {}).get("type")
        if method != "ircot":
            raise ValueError("--results-output currently supports only official IRCoT scoring")
        from eval.ircot_official_metrics import evaluate_predictions

        scored = {
            "metrics": evaluate_predictions(merged["results"].values()),
            "meta": {
                **(merged.get("metadata") or {}),
                "preds_path": os.path.abspath(args.output),
                "metric": "official_ircot_drop",
            },
            "inference_params": merged.get("inference_params") or {},
        }
        _atomic_write(args.results_output, scored)

    print(
        json.dumps(
            {
                "output": os.path.abspath(args.output),
                "results_output": os.path.abspath(args.results_output) if args.results_output else None,
                "count": len(merged["results"]),
                "shards": len(shard_records),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
