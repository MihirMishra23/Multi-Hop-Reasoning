#!/usr/bin/env python3
"""Prepare the exact HotpotQA parquets and retrieval corpus used by Search-R1."""

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.hotpotqa import load_hotpotqa  # noqa: E402


SYSTEM_PROMPT = "You are a helpful and harmless assistant."
USER_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and "
    "</think> first every time you get new information. After reasoning, if you "
    "find you lack some knowledge, you can call the search tool that is available "
    "to you; its results will be returned in a tool response. You can search as "
    "many times as you want. If you find no further external knowledge needed, "
    "directly provide the answer inside <answer> and </answer>, without detailed "
    "illustrations. For example, <answer> Beijing </answer>. Question: "
)


def _question(example):
    question = str(example["question"]).strip()
    return question if question.endswith("?") else question + "?"


def _verl_row(example, index, split):
    question = _question(example)
    targets = [str(answer) for answer in example.get("answers", [])] or [""]
    ground_truth = {"target": targets}
    data_source = "searchR1_hotpotqa"
    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PREFIX + question},
        ],
        "ability": "multi-hop-reasoning",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "index": index,
            "need_tools_kwargs": True,
            "question": question,
            "split": split,
            "tools_kwargs": {
                "search": {
                    "create_kwargs": {
                        "ground_truth": ground_truth,
                        "question": question,
                        "data_source": data_source,
                    }
                }
            },
        },
        "metadata": None,
    }


def _corpus_rows(train_dataset):
    seen = set()
    rows = []
    for example in train_dataset:
        for context in example.get("contexts", []):
            context = str(context).strip()
            if not context or context in seen:
                continue
            seen.add(context)
            title, separator, text = context.partition(": ")
            contents = f'"{title}"\n{text}' if separator else context
            rows.append({"id": str(len(rows)), "contents": contents})
    return rows


def _write_parquet(dataset, split, path):
    rows = [_verl_row(example, index, split) for index, example in enumerate(dataset)]
    Dataset.from_list(rows).to_parquet(str(path))
    print(f"Wrote {len(rows):,} rows to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--train-size", type=int, default=7000)
    parser.add_argument("--eval-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    train = load_hotpotqa(
        setting="distractor",
        split="train",
        source="hf",
        sub_split="train",
        limit=args.train_size,
        seed=args.seed,
    )
    validation = load_hotpotqa(
        setting="distractor",
        split="train",
        source="hf",
        sub_split="eval",
        limit=args.eval_size,
        seed=args.seed,
    )

    _write_parquet(train, "train", args.data_dir / "train_verl.parquet")
    _write_parquet(validation, "test", args.data_dir / "test_verl.parquet")

    corpus = _corpus_rows(train)
    corpus_path = args.data_dir / "hotpotqa_corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as stream:
        for row in corpus:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(corpus):,} passages to {corpus_path}")


if __name__ == "__main__":
    main()
