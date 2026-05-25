#!/usr/bin/env python3
"""
Prepare HotpotQA corpus for Search-R1 retrieval.

Loads training data using the same approach as grpo_train.py,
then builds a corpus from all contexts in those examples.

Creates a jsonl file where each line is:
{"id": "0", "contents": '"Title"\nText content...'}
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_dataset


def extract_contexts_from_examples(dataset):
    """Extract all unique contexts from dataset examples."""
    seen = set()
    corpus = []
    corpus_id = 0

    for example in dataset:
        # Get contexts from the example
        contexts = example.get("contexts", [])

        for context in contexts:
            # Each context is formatted as "Title: content"
            context = context.strip()
            if not context or context in seen:
                continue

            seen.add(context)

            # Parse title and content
            # Format: "Title: content text..."
            if ": " in context:
                title, content = context.split(": ", 1)
            else:
                title = ""
                content = context

            corpus.append({
                "id": corpus_id,
                "title": title,
                "contents": content
            })
            corpus_id += 1

    return corpus


def main():
    # Output: corpus in Search-R1 format
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    corpus_path = output_dir / "hotpotqa_corpus.jsonl"

    print("════════════════════════════════════════════════════════════════")
    print("HotpotQA Corpus Preparation for Search-R1")
    print("════════════════════════════════════════════════════════════════")
    print()

    # Load the SAME training examples that will be used in GRPO training
    # This uses sub_split="train" to get the last N examples (before the eval split)
    # matching exactly what grpo_train.sh does
    TRAIN_SIZE = 7000  # Same as TRAIN_SIZE in grpo_train.sh

    print(f"Loading HotpotQA training dataset (last {TRAIN_SIZE} examples, matching GRPO training)...")
    train_dataset = get_dataset(
        name="hotpotqa",
        setting="distractor",
        split="train",
        sub_split="train",  # Use the train sub-split (last N examples before eval)
        limit=TRAIN_SIZE,
        seed=42
    )
    print(f"✓ Loaded {len(train_dataset)} training examples (same as GRPO training)")

    # Extract all unique contexts from the dataset
    print("Extracting contexts from training examples...")
    corpus = extract_contexts_from_examples(train_dataset)
    print(f"✓ Extracted {len(corpus)} unique passages")

    # Write corpus in Search-R1 format
    print("Writing corpus in Search-R1 format...")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for record in corpus:
            # Search-R1 format: "contents" should be '"Title"\nText'
            title = record["title"]
            text = record["contents"]

            # Format as in Search-R1 examples
            if title:
                contents = f'"{title}"\n{text}'
            else:
                contents = text

            entry = {
                "id": str(record["id"]),
                "contents": contents
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✓ Wrote {len(corpus)} passages to {corpus_path}")

    # Print statistics
    total_chars = sum(len(r["contents"]) for r in corpus)
    avg_length = total_chars / len(corpus) if corpus else 0

    print()
    print("Corpus statistics:")
    print(f"  Total passages: {len(corpus):,}")
    print(f"  Average length: {avg_length:.0f} characters")
    print(f"  Total size:     {corpus_path.stat().st_size / 1024 / 1024:.1f} MB")
    print()
    print("✓ Corpus preparation complete!")
    print()


if __name__ == "__main__":
    main()
