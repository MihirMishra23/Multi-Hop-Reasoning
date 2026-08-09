#!/usr/bin/env python3
"""Materialize the reproducible 1,399-query PopQA long-tail evaluation inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from datasets import load_dataset  # type: ignore  # noqa: E402

from data.popqa import (  # noqa: E402
    LONG_TAIL_EXPECTED_COUNT,
    LONG_TAIL_MAX_SUBJECT_POPULARITY,
    _load_popqa_corpus,
    _resolve_popqa_corpus_path,
)
from data.provenance import (  # noqa: E402
    hf_source,
    selected_rows_provenance,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source = hf_source("popqa")
    raw = load_dataset(source["path"], split="test", revision=source["revision"])
    selected = raw.filter(
        lambda example: float(example["s_pop"]) < LONG_TAIL_MAX_SUBJECT_POPULARITY,
        desc="select PopQA long-tail subjects",
    )
    if len(selected) != LONG_TAIL_EXPECTED_COUNT:
        raise RuntimeError(
            f"Long-tail filter produced {len(selected)} rows; "
            f"expected {LONG_TAIL_EXPECTED_COUNT}"
        )
    selected = selected.shuffle(seed=args.seed)

    ordered_ids = [str(value) for value in selected["id"]]
    ordered_titles = [str(value).strip() for value in selected["s_wiki_title"]]
    unique_titles = list(dict.fromkeys(title for title in ordered_titles if title))

    full_corpus_path = _resolve_popqa_corpus_path(None)
    full_records = _load_popqa_corpus(full_corpus_path)
    by_title = {}
    for record in full_records:
        requested = str(record.get("requested_title") or "").strip()
        canonical = str(record.get("title") or "").strip()
        if requested:
            by_title.setdefault(requested, record)
        if canonical:
            by_title.setdefault(canonical, record)

    selected_records = []
    missing_titles = []
    seen_records = set()
    for title in unique_titles:
        record = by_title.get(title)
        if record is None:
            missing_titles.append(title)
            continue
        record_key = str(record.get("requested_title") or record.get("title") or title)
        if record_key in seen_records:
            continue
        seen_records.add(record_key)
        selected_records.append(record)

    corpus_path = output_dir / "popqa_longtail_contexts.jsonl"
    with corpus_path.open("w", encoding="utf-8") as stream:
        for record in selected_records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    selection = selected_rows_provenance(
        "popqa",
        ordered_ids,
        seed=args.seed,
        setting="long_tail",
    )
    manifest = {
        "schema_version": 1,
        "selection": selection,
        "filter": {
            "field": "s_pop",
            "operator": "<",
            "value": LONG_TAIL_MAX_SUBJECT_POPULARITY,
            "description": "subject monthly Wikipedia page views",
        },
        "corpus": {
            "source_path": str(full_corpus_path),
            "artifact": str(corpus_path),
            "sha256": sha256_file(corpus_path),
            "question_titles": len(ordered_titles),
            "unique_requested_titles": len(unique_titles),
            "records": len(selected_records),
            "usable_records": sum(
                record.get("status", "ok") == "ok" for record in selected_records
            ),
            "missing_titles": missing_titles,
        },
        "metric": {
            "name": "normalized_exact_match_any_alias",
            "normalization": "lowercase, remove punctuation/articles, normalize whitespace",
            "implementation": "src/eval/evaluate.py",
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "manifest": str(manifest_path),
        "ordered_ids_sha256": selection["selection"]["ordered_ids_sha256"],
        "questions": len(ordered_ids),
        "corpus_records": len(selected_records),
        "missing_titles": len(missing_titles),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
