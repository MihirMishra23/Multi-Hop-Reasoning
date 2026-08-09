# Search-R1 evaluation protocols

This repository has two deliberately separate Search-R1 inference paths.  A
checkpoint must be evaluated with the protocol it was trained with.

## Linxi Qwen3 ICL3Hop checkpoints (legacy Hermes protocol)

Use `eval_search_r1_hermes.py` for the merged checkpoints under
`rebuttal-search-r1/merged_ckpts/icl3hop_tr1024_step500` and
`rebuttal-search-r1/merged_ckpts/qwen3-4b-icl3hop-tr1024-step500`.  This file is
restored from commit `6c68b43`, the evaluator lineage that produced the saved
HotpotQA reference runs.  It uses:

- `<tool_call>{"name":"search",...}</tool_call>` and `<tool_response>` turns;
- prompt variant `icl3hop`, Qwen native thinking disabled;
- temperature 1.0, top-p 0.95, top-k 4, five turns;
- at most 2,048 generated tokens per turn and 1,024 characters per tool response;
- E5-base-v2 dense retrieval with top-k 3.

The Slurm launcher is `sbatch_eval_hermes.slurm`.  For a 50-question HotpotQA
smoke test using an already-built full dev corpus:

```bash
MERGED_CKPT=/path/to/merged_ckpt \
DATASET=hotpotqa SPLIT=dev NUM_SAMPLES=50 \
PROMPT_VARIANT=icl3hop ENABLE_THINKING=false \
MAX_TOOL_RESPONSE_LENGTH=1024 SAVE_FULL_OUTPUT=true \
CORPUS_PATH=/path/to/hotpotqa_dev/corpus.jsonl \
INDEX_PATH=/path/to/hotpotqa_dev/e5_Flat.index \
sbatch --partition=<authorized-partition> --gres=gpu:<available-gpu>:1 \
  rebuttal-search-r1/sbatch_eval_hermes.slurm
```

For ConFiQA, construct the retrieval store from all 1,000 ordered rows, even
when evaluating only the first 50 questions:

```bash
python rebuttal-search-r1/prepare_confiqa_corpus.py \
  --setting cf_356_conflict_free --num-samples 1000 --seed 42 \
  --output-dir /tmp/confiqa_cf356_store1000
python rebuttal-search-r1/build_retrieval_index.py \
  --corpus /tmp/confiqa_cf356_store1000/corpus.jsonl \
  --output /tmp/confiqa_cf356_store1000/e5_Flat.index

MERGED_CKPT=/path/to/icl3hop_tr1024_step500 \
DATASET=confiqa SPLIT=test NUM_SAMPLES=50 EXPECTED_STORE_SAMPLES=1000 \
CONFIQA_SETTING=cf_356_conflict_free \
CORPUS_MANIFEST=/tmp/confiqa_cf356_store1000/manifest.json \
CORPUS_PATH=/tmp/confiqa_cf356_store1000/corpus.jsonl \
INDEX_PATH=/tmp/confiqa_cf356_store1000/e5_Flat.index \
PROMPT_VARIANT=icl3hop ENABLE_THINKING=false \
MAX_TOOL_RESPONSE_LENGTH=1024 \
sbatch --partition=<authorized-partition> --gres=gpu:<available-gpu>:1 \
  rebuttal-search-r1/sbatch_eval_hermes.slurm
```

The evaluator rejects a ConFiQA store with the wrong condition, seed, size,
ordered-ID hash, or query-ID prefix, and embeds the manifest in its output.

## Released PeterJinGo Qwen2.5 checkpoints

Use the current integrated `SearchR1Agent` path in `src/eval_multihop.py` for
released `PeterJinGo/SearchR1-*` checkpoints.  That evaluator follows the
released `<search>`/`<information>` protocol.  Do not use it for the custom
Linxi ICL3Hop checkpoints.
