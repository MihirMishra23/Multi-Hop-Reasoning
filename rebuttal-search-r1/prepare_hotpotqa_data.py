#!/usr/bin/env python3
"""
Prepare HotpotQA data in Search-R1 parquet format.

Converts HotpotQA to the format expected by verl.trainer.main_ppo:
- prompt: list of dicts with role/content
- reward_model: dict with style and ground_truth
- data_source, ability, extra_info
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_dataset


def make_prefix(question: str) -> str:
    """Create the prompt prefix for Search-R1 style interaction."""
    prefix = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}
"""
    return prefix


def process_example(example, idx, split):
    """Convert HotpotQA example to Search-R1 format."""
    question = example['question'].strip()
    if question and question[-1] != '?':
        question += '?'

    prompt_text = make_prefix(question)

    # Get answers - HotpotQA has a list of answers
    answers = example.get('answers', [])
    if not answers:
        answers = ['']

    data = {
        "data_source": "hotpotqa",
        "prompt": [{
            "role": "user",
            "content": prompt_text,
        }],
        "ability": "multi-hop-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "target": answers  # List of acceptable answers
            }
        },
        "extra_info": {
            'split': split,
            'index': idx,
            'question_id': example.get('id', ''),
        }
    }
    return data


def main():
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    print("════════════════════════════════════════════════════════════════")
    print("HotpotQA Data Preparation for Search-R1 (Parquet Format)")
    print("════════════════════════════════════════════════════════════════")
    print()

    # Load training data (last 7k examples)
    TRAIN_SIZE = 7000
    EVAL_SIZE = 100

    print(f"Loading HotpotQA training data (last {TRAIN_SIZE} examples)...")
    train_dataset = get_dataset(
        name="hotpotqa",
        setting="distractor",
        split="train",
        sub_split="train",
        limit=TRAIN_SIZE,
        seed=42
    )
    print(f"✓ Loaded {len(train_dataset)} training examples")

    print(f"Loading HotpotQA validation data (last {EVAL_SIZE} examples)...")
    eval_dataset = get_dataset(
        name="hotpotqa",
        setting="distractor",
        split="train",
        sub_split="eval",
        limit=EVAL_SIZE,
        seed=42
    )
    print(f"✓ Loaded {len(eval_dataset)} validation examples")

    # Convert to Search-R1 format
    print("\nConverting to Search-R1 format...")

    def make_map_fn(split):
        def process_fn(example, idx):
            return process_example(example, idx, split)
        return process_fn

    train_dataset = train_dataset.map(
        function=make_map_fn('train'),
        with_indices=True,
        desc="Processing train data"
    )

    eval_dataset = eval_dataset.map(
        function=make_map_fn('test'),
        with_indices=True,
        desc="Processing eval data"
    )

    # Save as parquet
    train_path = output_dir / "train.parquet"
    eval_path = output_dir / "test.parquet"

    print(f"\nSaving to parquet format...")
    train_dataset.to_parquet(str(train_path))
    eval_dataset.to_parquet(str(eval_path))

    print(f"✓ Saved train data to {train_path}")
    print(f"✓ Saved eval data to {eval_path}")

    # Print statistics
    print()
    print("Data statistics:")
    print(f"  Training examples:   {len(train_dataset):,}")
    print(f"  Validation examples: {len(eval_dataset):,}")
    print(f"  Train file size:     {train_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Eval file size:      {eval_path.stat().st_size / 1024 / 1024:.1f} MB")
    print()
    print("✓ Data preparation complete!")
    print()


if __name__ == "__main__":
    main()
