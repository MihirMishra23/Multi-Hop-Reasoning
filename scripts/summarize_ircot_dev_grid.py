#!/usr/bin/env python3
"""Score and select the exact IRCoT prompt-set-1 development grid."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval.ircot_official_metrics import evaluate_predictions  # noqa: E402


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_root", help="Directory containing completed IRCoT grid JSON files")
    parser.add_argument("--output", required=True, help="Summary JSON path")
    parser.add_argument("--expected-count", type=int, default=100)
    args = parser.parse_args()

    rows = []
    pattern = os.path.join(args.prediction_root, "**", "*.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        with open(path, "r", encoding="utf-8") as source:
            payload = json.load(source)
        metadata = payload.get("metadata") or {}
        ircot = metadata.get("ircot") or {}
        results = payload.get("results") or {}
        if metadata.get("type") != "ircot" or ircot.get("prompt_set") != "1":
            continue
        if len(results) != args.expected_count:
            continue
        metrics = evaluate_predictions(results.values())
        rows.append(
            {
                "dataset": metadata["dataset"],
                "model": metadata["model"],
                "model_revision": metadata.get("model_revision"),
                "retrieval_k": int(ircot["retrieval_k"] if "retrieval_k" in ircot else metadata["retrieval"]["k"]),
                "distractor_count": int(ircot["distractor_count"]),
                "prompt_set": ircot["prompt_set"],
                **metrics,
                "prediction_path": path,
                "prediction_sha256": sha256_file(path),
                "evaluation_file_sha256": ircot.get("evaluation_file_sha256"),
                "evaluation_ids_sha256": ircot.get("evaluation_ids_sha256"),
                "prompt_sha256": ircot.get("prompt_sha256"),
                "index_manifest_sha256": ircot.get("index_manifest_sha256"),
                "code_commit": metadata.get("code_commit"),
            }
        )

    rows.sort(key=lambda row: (row["dataset"], row["model"], row["retrieval_k"], row["distractor_count"]))
    grouped = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["model"]), []).append(row)

    expected_grid = {(k, distractors) for k in (2, 4, 6, 8) for distractors in (1, 2, 3)}
    best = []
    for (dataset, model), group in sorted(grouped.items()):
        observed = {(row["retrieval_k"], row["distractor_count"]) for row in group}
        if observed != expected_grid:
            missing = sorted(expected_grid - observed)
            raise SystemExit(f"Incomplete grid for {dataset}/{model}; missing {missing}")
        # Upstream rounds F1 percentage to one decimal before comparing, then
        # iterates k first and distractor count second and replaces the current
        # best only on a strict increase.
        selected = None
        max_metric_value = float("-inf")
        for row in group:
            metric_value = round(row["f1"] * 100, 1)
            row["selection_f1_percent_rounded_1dp"] = metric_value
            if metric_value > max_metric_value:
                selected = row
                max_metric_value = metric_value
        assert selected is not None
        best.append(selected)

    summary = {
        "protocol": {
            "split": "released dev_subsampled.jsonl",
            "count": args.expected_count,
            "prompt_set": "1",
            "retrieval_k": [2, 4, 6, 8],
            "distractor_count": [1, 2, 3],
            "selection_metric": "upstream DROP F1 percentage rounded to one decimal",
        },
        "grid": rows,
        "best": best,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as destination:
        json.dump(summary, destination, ensure_ascii=False, indent=2)

    for row in rows:
        print(
            f"{row['dataset']:10s} {row['model']:11s} k={row['retrieval_k']} "
            f"d={row['distractor_count']} EM={100*row['em']:.1f} F1={100*row['f1']:.1f}"
        )
    for row in best:
        print(
            f"BEST {row['dataset']}/{row['model']}: k={row['retrieval_k']} "
            f"d={row['distractor_count']} EM={100*row['em']:.1f} F1={100*row['f1']:.1f}"
        )


if __name__ == "__main__":
    main()
