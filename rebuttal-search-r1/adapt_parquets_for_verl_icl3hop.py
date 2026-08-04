#!/usr/bin/env python3
"""
Variant of adapt_parquets_for_verl.py that uses an ICL-augmented user prompt:
- <thinking>/</thinking> tags (regular 3-token text) instead of <think>/</think>
  which are single special tokens in Qwen3 and collide with native thinking mode.
- A 3-hop search example demonstrating short single-sentence thinking + chained
  multi-turn search.

Output paths:
  data/train_verl_icl3hop.parquet
  data/test_verl_icl3hop.parquet
"""

import argparse
import json
from pathlib import Path
import pandas as pd

SYSTEM_CONTENT = "You are a helpful and harmless assistant."

# 3-hop ICL example. The example uses <thinking> (3 regular tokens) so it does
# not collide with Qwen3's <think> (single special token, reserved for native
# thinking mode). Each <thinking> block is one short sentence stating the next
# subgoal — sets a brevity expectation the model should imitate.
USER_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <thinking> "
    "and </thinking> first every time you get new information. Each <thinking> "
    "should be ONE BRIEF SENTENCE stating what you need next — not detailed "
    "analysis. After reasoning, if you lack knowledge, call the search tool. "
    "When you have enough information, provide the answer inside <answer> and "
    "</answer> without detailed illustrations.\n"
    "\n"
    "Here is an example showing the expected format and brevity:\n"
    "\n"
    "Question: What is the population of the capital city of the country where "
    "the inventor of the World Wide Web was born?\n"
    "\n"
    "<thinking>I need to find who invented the World Wide Web.</thinking>\n"
    '<tool_call>{"name": "search", "arguments": {"query_list": ["who invented the World Wide Web"]}}</tool_call>\n'
    '<tool_response>Doc 1 (Title: "Tim Berners-Lee"): Sir Timothy John Berners-Lee, '
    "also known as TimBL, is an English computer scientist best known as the "
    "inventor of the World Wide Web.</tool_response>\n"
    "<thinking>Tim Berners-Lee is English, so the country is the United Kingdom. "
    "I need the capital of the UK.</thinking>\n"
    '<tool_call>{"name": "search", "arguments": {"query_list": ["capital of the United Kingdom"]}}</tool_call>\n'
    '<tool_response>Doc 1 (Title: "London"): London is the capital and largest '
    "city of England and the United Kingdom.</tool_response>\n"
    "<thinking>The capital is London. Now I need the population of London."
    "</thinking>\n"
    '<tool_call>{"name": "search", "arguments": {"query_list": ["population of London"]}}</tool_call>\n'
    '<tool_response>Doc 1 (Title: "London"): London has a population of '
    "approximately 9 million people.</tool_response>\n"
    "<thinking>The population is about 9 million.</thinking>\n"
    "<answer>9 million</answer>\n"
    "\n"
    "Now answer this question: "
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
    print(json.dumps(rows[0], default=str, indent=2)[:1500])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=str(Path(__file__).parent / "data"))
    args = p.parse_args()
    d = Path(args.data_dir)
    convert(d / "train.parquet", d / "train_verl_icl3hop.parquet", split="train")
    convert(d / "test.parquet", d / "test_verl_icl3hop.parquet", split="test")


if __name__ == "__main__":
    main()
