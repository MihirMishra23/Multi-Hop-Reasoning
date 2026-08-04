#!/usr/bin/env python3
"""
Variant of adapt_parquets_for_verl.py that ONLY switches <think>/</think> to
<thinking>/</thinking> to avoid the Qwen3 special-token collision. No ICL
example added (clean ablation vs adapt_parquets_for_verl_icl3hop.py).

Output paths:
  data/train_verl_thinkingtag.parquet
  data/test_verl_thinkingtag.parquet
"""

import argparse
import json
from pathlib import Path
import pandas as pd

SYSTEM_CONTENT = "You are a helpful and harmless assistant."

USER_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <thinking> "
    "and </thinking> first every time you get new information. After reasoning, "
    "if you find you lack some knowledge, you can call the search tool that is "
    "available to you; its results will be returned in a tool response. You can "
    "search as many times as you want. If you find no further external knowledge "
    "needed, directly provide the answer inside <answer> and </answer>, without "
    "detailed illustrations. For example, <answer> Beijing </answer>. Question: "
)

_GARBAGE_SENTINELS = (
    "\\n',", "\\n\",",
    "\n',", "\n\",",
    "', 'role'", "\", \"role\"",
    "'}]", "\"}]",
)


def adapt_row(row, split, idx):
    old_prompt = row["prompt"]
    if isinstance(old_prompt, list) and old_prompt and isinstance(old_prompt[0], dict):
        old_user = old_prompt[0].get("content", "")
    else:
        old_user = str(old_prompt)
    q_marker = "Question: "
    if q_marker in old_user:
        question = old_user.split(q_marker, 1)[1]
    else:
        question = old_user
    for sentinel in _GARBAGE_SENTINELS:
        if sentinel in question:
            question = question.split(sentinel, 1)[0]
            break
    question = question.rstrip("\n").strip()

    user_content = USER_PREFIX + question
    prompt = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": user_content},
    ]

    reward_model = row.get("reward_model")
    if not isinstance(reward_model, dict):
        reward_model = {"style": "rule", "ground_truth": {"target": []}}
    ground_truth = reward_model.get("ground_truth", {"target": []})

    data_source_tagged = "searchR1_" + str(row.get("data_source", "hotpotqa"))

    extra_info = {
        "index": idx,
        "need_tools_kwargs": True,
        "question": question,
        "split": split,
        "tools_kwargs": {
            "search": {
                "create_kwargs": {
                    "ground_truth": ground_truth,
                    "question": question,
                    "data_source": data_source_tagged,
                }
            }
        },
    }

    return {
        "data_source": data_source_tagged,
        "prompt": prompt,
        "ability": row.get("ability"),
        "reward_model": reward_model,
        "extra_info": extra_info,
        "metadata": row.get("metadata") if "metadata" in row else None,
    }


def convert(src, dst, split):
    df = pd.read_parquet(src)
    rows = [adapt_row(r._asdict() if hasattr(r, "_asdict") else r, split, i)
            for i, r in enumerate(df.to_dict(orient="records"))]
    out = pd.DataFrame(rows)
    out.to_parquet(dst)
    print(f"✓ {src} → {dst} ({len(out)} rows)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=str(Path(__file__).parent / "data"))
    args = p.parse_args()
    d = Path(args.data_dir)
    convert(d / "train.parquet", d / "train_verl_thinkingtag.parquet", split="train")
    convert(d / "test.parquet", d / "test_verl_thinkingtag.parquet", split="test")


if __name__ == "__main__":
    main()
