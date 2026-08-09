#!/usr/bin/env python3
"""Validate and aggregate the exact IRCoT best-HP test evaluation."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval.ircot_official_metrics import evaluate_predictions  # noqa: E402


SELECTED = {
    ("hotpotqa", "qwen3-1.7b"): (8, 1),
    ("hotpotqa", "qwen3-4b"): (6, 1),
    ("2wiki", "qwen3-1.7b"): (4, 3),
    ("2wiki", "qwen3-4b"): (6, 1),
    ("musique", "qwen3-1.7b"): (4, 2),
    ("musique", "qwen3-4b"): (2, 2),
}


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-count", type=int, default=500)
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.prediction_root, "**", "*.json"), recursive=True)):
        with open(path, "r", encoding="utf-8") as source:
            payload = json.load(source)
        metadata = payload.get("metadata") or {}
        ircot = metadata.get("ircot") or {}
        results = payload.get("results") or {}
        if metadata.get("type") != "ircot" or metadata.get("split") != "test":
            continue
        if len(results) != args.expected_count:
            continue
        errors = [
            (qid, step.get("error"))
            for qid, result in results.items()
            for step in (result.get("trace") or [])
            if step.get("error")
        ]
        if errors:
            raise SystemExit(f"Invalid artifact {path}: {len(errors)} trace errors; first={errors[0]}")
        dataset = metadata["dataset"]
        model = metadata["model"]
        key = (dataset, model)
        if key not in SELECTED:
            raise SystemExit(f"Unexpected test group in {path}: {key}")
        retrieval_k = int(ircot["retrieval_k"])
        distractors = int(ircot["distractor_count"])
        if (retrieval_k, distractors) != SELECTED[key]:
            raise SystemExit(
                f"Wrong selected setting for {dataset}/{model}: "
                f"{(retrieval_k, distractors)} != {SELECTED[key]}"
            )
        prompt_set = str(ircot["prompt_set"])
        if prompt_set not in {"1", "2", "3"}:
            raise SystemExit(f"Unexpected prompt set in {path}: {prompt_set}")
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "retrieval_k": retrieval_k,
                "distractor_count": distractors,
                "prompt_set": prompt_set,
                **evaluate_predictions(results.values()),
                "prediction_path": path,
                "prediction_sha256": sha256_file(path),
                "evaluation_file_sha256": ircot.get("evaluation_file_sha256"),
                "evaluation_ids_sha256": ircot.get("evaluation_ids_sha256"),
                "prompt_sha256": ircot.get("prompt_sha256"),
                "index_manifest_sha256": ircot.get("index_manifest_sha256"),
                "model_revision": metadata.get("model_revision"),
                "code_commit": metadata.get("code_commit"),
            }
        )

    rows.sort(key=lambda row: (row["dataset"], row["model"], row["prompt_set"]))
    expected_cells = {(dataset, model, prompt_set) for dataset, model in SELECTED for prompt_set in ("1", "2", "3")}
    observed_cells = {(row["dataset"], row["model"], row["prompt_set"]) for row in rows}
    if observed_cells != expected_cells or len(rows) != len(expected_cells):
        raise SystemExit(
            f"Incomplete or duplicate test grid: missing={sorted(expected_cells - observed_cells)} "
            f"extra={sorted(observed_cells - expected_cells)} rows={len(rows)}"
        )

    aggregate = []
    for dataset, model in sorted(SELECTED):
        group = [row for row in rows if row["dataset"] == dataset and row["model"] == model]
        for field in ("evaluation_file_sha256", "evaluation_ids_sha256", "model_revision", "code_commit"):
            if len({row[field] for row in group}) != 1:
                raise SystemExit(f"Inconsistent {field} for {dataset}/{model}")
        em_values = [row["em"] for row in group]
        f1_values = [row["f1"] for row in group]
        retrieval_k, distractors = SELECTED[(dataset, model)]
        aggregate.append(
            {
                "dataset": dataset,
                "model": model,
                "retrieval_k": retrieval_k,
                "distractor_count": distractors,
                "prompt_sets": ["1", "2", "3"],
                "em_mean": statistics.mean(em_values),
                "em_sample_std": statistics.stdev(em_values),
                "f1_mean": statistics.mean(f1_values),
                "f1_sample_std": statistics.stdev(f1_values),
                "evaluation_file_sha256": group[0]["evaluation_file_sha256"],
                "evaluation_ids_sha256": group[0]["evaluation_ids_sha256"],
                "model_revision": group[0]["model_revision"],
                "code_commit": group[0]["code_commit"],
            }
        )

    summary = {
        "protocol": {
            "split": "released test_subsampled.jsonl",
            "count": args.expected_count,
            "prompt_sets": ["1", "2", "3"],
            "aggregation": "mean and sample standard deviation across prompt sets",
            "selected_on": "released 100-row dev_subsampled.jsonl prompt set 1",
        },
        "prompt_results": rows,
        "aggregate": aggregate,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as destination:
        json.dump(summary, destination, ensure_ascii=False, indent=2)

    for row in rows:
        print(
            f"{row['dataset']:10s} {row['model']:11s} ps={row['prompt_set']} "
            f"EM={100 * row['em']:.1f} F1={100 * row['f1']:.1f}"
        )
    for row in aggregate:
        print(
            f"AGG {row['dataset']}/{row['model']}: k={row['retrieval_k']} "
            f"d={row['distractor_count']} EM={100 * row['em_mean']:.1f}±{100 * row['em_sample_std']:.1f} "
            f"F1={100 * row['f1_mean']:.1f}±{100 * row['f1_sample_std']:.1f}"
        )


if __name__ == "__main__":
    main()
