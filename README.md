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
- Agent API: `src/agent/` (`agent.py`, `icl_agent.py`, `rag_agent.py`)
- LLM adapters: `src/llm/` (`base.py`, `openai.py`, `llama.py`)
- Datasets: `src/data/` (`hotpotqa.py`, `musique.py`, and `get_dataset` dispatcher)
- Evaluation: `src/eval/` (pure functions) and `scripts/evaluate.py` (CLI)

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

3) Install FlashRAG:

```bash
cd src/tools
git clone https://github.com/RUC-NLPIR/FlashRAG.git
cd FlashRAG
pip install -e .
```

4) Set credentials (OpenAI):

```bash
export OPENAI_API_KEY="<your-key>"
# Optional: export OPENAI_BASE_URL="https://api.openai.com/v1"
```

5) Generate predictions (writes to `preds/<method>/<dataset>_<setting>/<model>/...`):

```bash
python scripts/eval_multihop.py \
  --dataset hotpotqa \
  --setting distractor \
  --split validation \
  --method icl \
  --model gpt-4 \
  --start_idx \
  --batch-size 10
```

RAG examples:

```bash
# Per-example distractor RAG (current behavior)
python scripts/eval_multihop.py \
  --dataset hotpotqa \
  --setting distractor \
  --split validation \
  --method rag \
  --model gpt-4
```

```bash
# FullWiki RAG (defaults to fullwiki corpus path when setting=fullwiki)
python scripts/eval_multihop.py \
  --dataset hotpotqa \
  --setting fullwiki \
  --split validation \
  --method rag \
  --model gpt-4 \
  --rag-corpus-path /share/j_sun/lmlm_multihop/datasets/hotpot_dev_fullwiki_v1.json
```

This will produce a file like:

```
preds/icl/hotpotqa_distractor/gpt-4/validation_seed=0_bn=1_bs=10.json
```

The script supports batch processing:
- `--batch-number`: Starting batch number (1-based)
- `--batch-size`: Number of examples per batch
- `--num-batches`: Number of batches to process (use `-1` to process all remaining batches)
- `--resume`: Skip batches that already exist (useful for resuming interrupted runs)

6) Evaluate predictions (writes a timestamped JSON to `results/eval`):

The evaluator supports three modes:

**Single file:**
```bash
python scripts/evaluate.py \
  --preds preds/icl/hotpotqa_distractor/gpt-4/validation_seed=0_bn=1_bs=10.json \
  --outdir results/eval
```

**Pattern-based (evaluate multiple batch files):**
```bash
python scripts/evaluate.py \
  --pattern "preds/icl/hotpotqa_distractor/gpt-4/validation_seed=0_bn=*_bs=10.json" \
  --outdir results/eval
```

**Directory-based:**
```bash
python scripts/evaluate.py \
  --input-dir preds/icl/hotpotqa_distractor/gpt-4 \
  --split validation \
  --seed 0 \
  --batch-size 10 \
  --outdir results/eval
```

### Unified prediction format (repo-wide)

Predictions saved under `preds/` use a consistent JSON layout with deduplicated metadata at the top level:

```json
{
  "metadata": {
    "model": "gpt-4",
    "split": "validation",
    "batch_size": 10,
    "batch_number": 1,
    "type": "icl",
    "seed": 0
  },
  "inference_params": {
    "seed": 0,
    "temperature": 0.0,
    "max_tokens": 256
  },
  "results": {
    "<qid>": {
      "pred": "answer string",
      "gold_answer": ["gold1", "gold2"],
      "gold_evidence": [...],
      "question": "original question",
      "trace": [...]
    }
  }
}
```

Notes:
- Metadata is deduplicated at the top level to reduce file size
- Additional fields like `trace`, `gold_evidence`, or `evidence` may be present depending on the generator and method
- The evaluator looks for `pred` and a gold field (`gold_answer` or `answers`) and computes EM/F1/precision/recall accordingly
- For RAG method, `metadata` may include a `retrieval` field with backend, scope, and k parameters

### Evaluation details

The evaluator supports three input modes:

1. **Single file** (`--preds`): Evaluate one prediction file
2. **Pattern-based** (`--pattern`): Evaluate multiple batch files matching a glob pattern (e.g., `*_bn=*_bs=10.json`)
3. **Directory-based** (`--input-dir`): Evaluate all batch files in a directory matching specified split/seed/batch-size

When evaluating multiple files, they are automatically aggregated before evaluation. The output filename includes question ranges for multi-file evaluations (e.g., `01-15_validation_seed=0_bs=10_q1-150_evaluation.json`).

The evaluator can fetch gold answers from Hugging Face when they are not embedded in the preds file:

```bash
python scripts/evaluate.py \
  --preds preds/icl/hotpotqa_distractor/gpt-4/validation_seed=0_bn=1_bs=10.json \
  --outdir results/eval \
  --source hf
```

Flags:
- `--dataset/--setting/--split` can override values parsed from filename or metadata
- `--source` chooses where to fetch gold labels: `hf` (Hugging Face) or `local` if you maintain raw JSONs under `data/raw/hotpotqa`
- `--agent/--llm` can override agent/model names in metadata

The output JSON written under `results/eval/` includes:
- `metrics`: `count`, `em`, `f1`, `precision`, `recall`
- `meta`: run metadata (dataset, agent, llm, bn, bs, split, timestamp, preds_path)

### Datasets & resources

- HotpotQA (Hugging Face): `https://huggingface.co/datasets/hotpot_qa`
- MuSiQue (Hugging Face): `https://huggingface.co/datasets/allenai/musique`
