## Limited Memory Language Models with Multi-Hop Reasoning

### Overview

This repository provides a lightweight toolkit for experimenting with limited-memory LLMs that perform multi-hop reasoning. It includes:
- An extensible agent loop for iterative reasoning with a constrained memory trace
- Pluggable LLM providers (OpenAI out of the box)
- Unified dataset loaders for multi-hop QA (HotpotQA, MuSiQue)
- Evaluation utilities for answer-only metrics (EM/F1/precision/recall)
- Ready-to-run scripts and examples for generating predictions and evaluating them

Supported datasets:
- HotpotQA (distractor/fullwiki) — see `src/data/hotpotqa.py` and Hugging Face `https://huggingface.co/datasets/hotpot_qa`
- MuSiQue — see `src/data/musique.py` and Hugging Face `https://huggingface.co/datasets/allenai/musique`

### Method and approach

The agent loop maintains a compact trace (limited memory) across steps:
1) Build a prompt from the current question and the short trace
2) Query an LLM and parse the action
3) Continue generation or finish when a final answer is produced

The base `Agent` supports simple generate/finish behaviors and can be extended to support tool use (e.g., retrieval/search). Evaluation focuses on answer-only metrics and can optionally stream gold labels from datasets for convenience.

### Key features
- Agent API: `src/agent/agent.py`
- LLM adapters: `src/llm/` (`base.py`, `openai.py`)
- Datasets: `src/data/` (`hotpotqa.py`, `musique.py`, and `get_dataset` dispatcher)
- Evaluation: `src/eval/` (pure functions) and `scripts/evaluate.py` (CLI)
- Examples: `examples/` (e.g., HotpotQA/MuSiQue prompting scripts)

### Repository structure (selected)

```
src/
  agent/agent.py        # Minimal agent loop with limited memory trace
  llm/base.py           # Provider-agnostic LLM interface
  llm/openai.py         # OpenAI Responses API adapter
  data/                 # Dataset loaders (HotpotQA, MuSiQue)
  eval/                 # Answer-only metrics and evaluation helpers
scripts/
  run_agent.py          # Generate predictions (writes to preds/...)
  evaluate.py           # Evaluate a preds JSON (writes to results/eval)
preds/                  # Saved predictions (by agent/setting)
results/                # Evaluation outputs
```

### Installation

See `INSTALLATION.md` for full details, including editable install and environment setup (`OPENAI_API_KEY`).

### Quickstart (scripts)

1) Create and activate a conda environment (Python 3.12):

```bash
conda create -n lmlm python=3.12 -y
conda activate lmlm
```

2) Install the package in editable mode:

```bash
pip install -e .
```

3) Set credentials (OpenAI):

```bash
export OPENAI_API_KEY="<your-key>"
# Optional: export OPENAI_BASE_URL="https://api.openai.com/v1"
```

4) Generate predictions (writes to `preds/<method>/...`):

```bash
python scripts/run_agent.py \
  --dataset hotpotqa \
  --setting distractor \
  --split validation \
  --method icl \
  --batch-number 1 \
  --batch-size 1
```

This will produce a file like:

```
preds/icl/hotpotqa_distractor_validation_bn=1_bs=1.json
```

5) Evaluate predictions (writes a timestamped JSON to `results/eval`):

```bash
python scripts/evaluate.py \
  --preds preds/icl/hotpotqa_distractor_validation_bn=1_bs=1.json \
  --outdir results/eval
```

### Unified prediction format (repo-wide)

Predictions saved under `preds/` use a consistent JSON layout across datasets (object keyed by example id). Minimal example:

```json
{
  "<qid>": {
    "pred": "answer string",
    "gold_answer": ["gold1", "gold2"],
    "question": "original question",
    "metadata": {
      "model": "gpt-4",
      "split": "validation",
      "batch_number": 1,
      "batch_size": 1,
      "type": "icl"
    },
    "inference_params": {
      "seed": 0,
      "temperature": 0.0,
      "max_tokens": 256
    }
  }
}
```

Notes:
- Additional fields like `trace` or `gold_evidence` may be present depending on the generator.
- The evaluator looks for `pred` and a gold field (`gold_answer` or `answers`) and computes EM/F1/precision/recall accordingly.

### HotpotQA evaluation details

Use the provided CLI to evaluate any `preds/...json` file. The evaluator can fetch gold answers from Hugging Face when they are not embedded in the preds file.

```bash
python scripts/evaluate.py \
  --preds preds/icl/hotpotqa_distractor_validation_bn=1_bs=1.json \
  --outdir results/eval \
  --source hf
```

Flags:
- `--dataset/--setting/--split` can override values parsed from filename or metadata
- `--source` chooses where to fetch gold labels: `hf` (Hugging Face) or `local` if you maintain raw JSONs under `data/raw/hotpotqa`

The output JSON written under `results/eval/` includes:
- `metrics`: `count`, `em`, `f1`, `precision`, `recall`
- `meta`: run metadata (dataset, agent, llm, bn, bs, split, timestamp)

### Datasets & resources

- HotpotQA (Hugging Face): `https://huggingface.co/datasets/hotpot_qa`
- MuSiQue (Hugging Face): `https://huggingface.co/datasets/allenai/musique`