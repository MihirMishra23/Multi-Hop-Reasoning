# HotpotQA Evaluation Quick Guide

## Setup
- `pip install -r /home/mrm367/Multi-Hop-Reasoning/requirements.txt`
- Optional: create a virtualenv beforehand (`python -m venv .venv && source .venv/bin/activate`).

## Get HotpotQA Data
- Online: the evaluator streams examples via `datasets.load_dataset("hotpot_qa", "distractor")`; a Hugging Face token is not required for public access.
- Offline: download `hotpot_train_v1.1.json` and `hotpot_dev_distractor_v1.json` from <https://hotpotqa.github.io/> and expose them with `--data-path /abs/path/to/jsons`.

## Prediction Format
- JSON object with two top-level maps, `answer` and `sp`.
- `answer[id]` is the final prediction string for that HotpotQA example.
- `sp[id]` is a list of `[page_title, sentence_index]` pairs identifying evidence sentences (index is 0-based within the page).
- Example:

```
{
  "answer": {
    "5a8b57f25542995d1e6f1371": "yes"
  },
  "sp": {
    "5a8b57f25542995d1e6f1371": [["Scott Derrickson", 0], ["Ed Wood", 0]]
  }
}
```

