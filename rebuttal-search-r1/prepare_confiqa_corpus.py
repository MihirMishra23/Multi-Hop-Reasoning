#!/usr/bin/env python3
"""Create the exact ConFiQA retrieval corpus used by a Search-R1 eval run."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import (  # noqa: E402
    conflict_free_condition_metadata,
    get_dataset,
    selected_rows_provenance,
)
from data.provenance import sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setting",
        choices=[
            "orig",
            "cf",
            "cf_100",
            "cf_500",
            "cf_100_conflict_free",
            "cf_356_conflict_free",
        ],
        required=True,
    )
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    dataset = get_dataset(
        name="confiqa",
        setting=args.setting,
        split="test",
        source="auto",
        limit=args.num_samples,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.jsonl"

    with open(corpus_path, "w", encoding="utf-8") as stream:
        for row in dataset:
            contexts = row.get("contexts") or []
            for context_index, context in enumerate(contexts):
                record_id = f"{row['id']}:{context_index}"
                record = {
                    "id": record_id,
                    "contents": f'"ConFiQA {row["id"]}"\n{context}',
                    "source_example_id": row["id"],
                    "is_counterfactual": row["is_counterfactual"],
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    counterfactual_count = sum(bool(value) for value in dataset["is_counterfactual"])
    provenance = selected_rows_provenance(
        "confiqa",
        dataset["id"],
        seed=args.seed,
        setting=args.setting,
        counterfactual_count=counterfactual_count,
    )
    manifest = {
        "dataset_provenance": provenance,
        "conflict_free_condition": conflict_free_condition_metadata(args.setting),
        "corpus": {
            "path": corpus_path.name,
            "sha256": sha256_file(corpus_path),
            "passages": sum(len(row.get("contexts") or []) for row in dataset),
        },
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {"corpus": str(corpus_path), "manifest": str(manifest_path), **manifest["corpus"]}
        )
    )


if __name__ == "__main__":
    main()
