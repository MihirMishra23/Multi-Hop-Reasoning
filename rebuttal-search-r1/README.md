# Search-R1 Qwen3 paper-run reproduction

This directory reproduces Linxi's May 2026 Search-R1 training path from
`c30a5d9`: prepare the 7,000/100 HotpotQA split, build the E5-base-v2 FAISS
retriever, start the verl-compatible retrieval service, and train Qwen3 with
the recorded GRPO recipe.

The later tuning in `522b9ba` is intentionally not used because it changed the
paper-run hyperparameters.

## Run from a clean checkout

```bash
conda env create -f rebuttal-search-r1/environment.yml
conda activate searchr1-reproduction
wandb login

# Qwen3-1.7B, four GPUs
bash rebuttal-search-r1/reproduce_training.sh

# Qwen3-4B
MODEL_SIZE=4B bash rebuttal-search-r1/reproduce_training.sh
```

The first invocation downloads HotpotQA and the E5/Qwen models, writes generated
artifacts under `rebuttal-search-r1/data/`, builds the index, launches retrieval
on port 8000, and then starts training. The retriever is stopped when training
exits.

The historical defaults use four GPUs, 7,000 training prompts, 100 validation
prompts, five rollouts per prompt, 500 steps, and W&B project
`search-r1-hotpotqa`. Override `WANDB_ENTITY`, `NUM_GPUS`, `LR`, `N_ROLLOUT`,
or pass trailing Hydra overrides to `reproduce_training.sh` when needed.

Useful overrides:

```bash
# Build the index or run retrieval on CPU (slower, but saves training VRAM)
INDEX_DEVICE=cpu RETRIEVER_DEVICE=cpu bash rebuttal-search-r1/reproduce_training.sh

# Rebuild generated artifacts
FORCE_PREPARE=1 FORCE_INDEX=1 bash rebuttal-search-r1/reproduce_training.sh

# Reuse a retriever that you deliberately started yourself
REUSE_RETRIEVER=1 bash rebuttal-search-r1/reproduce_training.sh
```

The recorded runs used verl 0.7, SGLang 0.5.2, PyTorch 2.8/CUDA 12.8, and
Transformers 4.56.1 on four B200 GPUs. Full training cannot be validated on a
CPU-only machine.
