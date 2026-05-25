#!/usr/bin/env python3
"""
Convert the parquets produced by prepare_hotpotqa_data.py into the schema
verl's search_r1_like recipe expects (per
verl/examples/data_preprocess/preprocess_search_r1_dataset.py).

Per-row schema verl wants:
  data_source       : "searchR1_hotpotqa"
  prompt            : [{"role":"system","content":...},{"role":"user","content":...}]
  ability           : (passthrough)
  reward_model      : {"style":"rule","ground_truth":{"target":[...answers...]}}
  extra_info        : {
      "index": int, "need_tools_kwargs": True, "question": str, "split": "train"|"test",
      "tools_kwargs": {"search": {"create_kwargs": {
          "ground_truth": <ground_truth>, "question": str, "data_source": str,
      }}}
  }
  metadata          : (passthrough or None)

Our existing rows already have data_source/prompt/ability/reward_model/extra_info,
but the prompt is single-turn (user only, no system) and extra_info has no
tools_kwargs. We rewrite both.
"""

import argparse
import json
from pathlib import Path
import pandas as pd

SYSTEM_CONTENT = "You are a helpful and harmless assistant."
USER_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <tool_call> query </tool_call> "
    "and it will return the top searched results between <tool_response> and "
    "</tool_response>. You can search as many times as your want. If you find no "
    "further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>. Question: "
)


def adapt_row(row, split, idx):
    # Pull question — original prompt[0]["content"] starts with our old prefix; strip it
    old_prompt = row["prompt"]
    if isinstance(old_prompt, list) and old_prompt and isinstance(old_prompt[0], dict):
        old_user = old_prompt[0].get("content", "")
    else:
        old_user = str(old_prompt)
    # Extract just the question part: "...Question: <q>\n"
    q_marker = "Question: "
    if q_marker in old_user:
        question = old_user.split(q_marker, 1)[1].rstrip("\n").rstrip()
    else:
        question = old_user.strip()

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
    # Sanity dump
    print(json.dumps(rows[0], default=str, indent=2)[:800])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=str(Path(__file__).parent / "data"))
    args = p.parse_args()
    d = Path(args.data_dir)
    convert(d / "train.parquet", d / "train_verl.parquet", split="train")
    convert(d / "test.parquet", d / "test_verl.parquet", split="test")


if __name__ == "__main__":
    main()
