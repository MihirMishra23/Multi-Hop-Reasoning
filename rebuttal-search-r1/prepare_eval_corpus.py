#!/usr/bin/env python3
"""
Build an eval-only retrieval corpus for a single benchmark.

For Search-R1 eval, we want a retrieval datastore that contains ONLY the
documents present in this benchmark's eval set (1000 questions). This makes
the retrieval recall a function of question difficulty + retrieval quality,
not of corpus coverage. The user prefers per-benchmark eval-only corpora.

Output: one JSONL per (benchmark, split), each line is
    {"id": <int>, "title": <str>, "content": <str>}

Usage:
    python3 prepare_eval_corpus.py --benchmark hotpotqa --split train_val1k --n_samples 1000
    python3 prepare_eval_corpus.py --benchmark musique --split dev --n_samples 1000
    python3 prepare_eval_corpus.py --benchmark 2wiki --split dev --n_samples 1000
    python3 prepare_eval_corpus.py --benchmark popqa --split dev --n_samples 1000
    python3 prepare_eval_corpus.py --benchmark hotpotqa --split dev --n_samples 1000
"""

import argparse
import json
import sys
from pathlib import Path

# Make src importable (parent of this file).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import get_dataset


# Mirror the (dataset, split_label) → (setting, real_split, start_idx) mapping
# used by eval_search_r1.py (line ~158) so the corpus we build matches the
# exact eval window the model will be queried on.
DATASET_SPLIT_MAP = {
    ("hotpotqa",  "dev"):           ("distractor", "validation", 0),
    ("hotpotqa",  "train_val1k"):   ("distractor", "train",      82347),
    ("hotpotqa",  "train_train1k"): ("distractor", "train",      89347),
    ("musique",   "dev"):           (None,         "validation", 0),
    ("2wiki",     "dev"):           ("distractor", "validation", 0),
    ("trivia_qa", "dev"):           (None,         "validation", 0),
    ("popqa",     "dev"):           (None,         "test",       0),
}


def extract_contexts(ds, n_samples):
    """Extract unique contexts. Output schema matches data/hotpotqa_corpus.jsonl:
        {"id": "<int_as_str>", "contents": '"<title>"\\n<content>'}
    """
    seen, corpus, cid = set(), [], 0
    for ex in ds.select(range(min(n_samples, len(ds)))):
        ctxs = ex.get("contexts") or []
        for ctx in ctxs:
            ctx = (ctx or "").strip()
            if not ctx or ctx in seen:
                continue
            seen.add(ctx)
            if ": " in ctx:
                title, content = ctx.split(": ", 1)
            else:
                title, content = "", ctx
            # Combine into the `contents` field, mirroring data/hotpotqa_corpus.jsonl
            contents = f'"{title}"\n{content}'
            corpus.append({"id": str(cid), "contents": contents})
            cid += 1
    return corpus


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True,
                   choices=["hotpotqa", "musique", "2wiki", "trivia_qa", "popqa"])
    p.add_argument("--split", required=True,
                   help="Label split, e.g. dev / train_val1k (will be mapped to real split)")
    p.add_argument("--n_samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "data" / "eval_corpora"))
    args = p.parse_args()

    key = (args.benchmark, args.split)
    if key not in DATASET_SPLIT_MAP:
        raise ValueError(f"Unsupported (benchmark, split)={key}; allowed: {list(DATASET_SPLIT_MAP)}")
    setting, real_split, start_idx = DATASET_SPLIT_MAP[key]

    print(f"Loading {args.benchmark} setting={setting} split={real_split} ({args.split}) "
          f"limit={start_idx + args.n_samples}")
    ds = get_dataset(name=args.benchmark, setting=setting, split=real_split, seed=args.seed)

    # Slice to the same window eval will use.
    end_idx = min(start_idx + args.n_samples, len(ds))
    ds = ds.select(range(start_idx, end_idx))
    print(f"  loaded {len(ds)} questions")

    corpus = extract_contexts(ds, len(ds))
    print(f"  extracted {len(corpus)} unique passages")

    out_subdir = Path(args.out_dir) / f"{args.benchmark}_{args.split}"
    out_subdir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_subdir / "corpus.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in corpus:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {out_jsonl}")


if __name__ == "__main__":
    main()
