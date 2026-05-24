# Search-R1 Training for HotpotQA with Qwen3

Train Qwen3 models on multi-hop question answering using reinforcement learning.

## Requirements

- **Hardware**: 8x NVIDIA GPUs with ~80GB+ VRAM each (tested on B200)
- **Software**: CUDA 12.1+, Python 3.10, Conda

## Setup (5 steps)

### 1. Install Environment

```bash
# Option A: Exact reproduction (recommended)
pip install conda-lock
conda-lock install --name searchr1 conda-lock.yml

# Option B: From environment.yml
conda env create -f environment.yml

# Activate and install Search-R1
conda activate searchr1
bash setup_environment.sh
```

### 2. Prepare Data

```bash
python prepare_hotpotqa_corpus.py  # Creates corpus
python prepare_hotpotqa_data.py    # Creates train/test splits
bash build_index.sh                 # Builds retrieval index
```

### 3. Start Retrieval Server (separate terminal)

```bash
python retrieval_server.py \
    --index_path=data/index/e5_Flat.index \
    --corpus_path=data/hotpotqa_corpus.jsonl \
    --port=8000
```

### 4. Configure WandB (optional)

```bash
wandb login
# Edit train.sh line 32: set your WANDB_ENTITY
```

### 5. Train

```bash
# Ensure using the correct Python environment
export PATH="/scratch/rtn27/envs/searchr1/bin:$PATH"

# Qwen3-1.7B on 8 GPUs
./train.sh --model_size 1.7B --num_gpus 8

# Qwen3-4B on 8 GPUs
./train.sh --model_size 4B --num_gpus 8
```

## Key Details

### Models
- **Qwen3-1.7B** or **Qwen3-4B** (base models from HuggingFace)
- Downloads automatically if not cached

### Hyperparameters (Search-R1 paper defaults)
- Batch size: 512 (distributed across 8 GPUs)
- Training steps: 500
- Rollouts per question: 32
- Learning rate: 1e-6
- Reward: F1 score (not exact match)

### Qwen3 Integration
Qwen3 support added to vLLM 0.6.3 via custom implementation:
1. Created `qwen3.py` model file with proper Qwen3 architecture (q_norm/k_norm layers)
2. Registered `Qwen3ForCausalLM` in model registry
3. Added Qwen3 config mapping to vLLM's config loader

These patches are applied to `/scratch/rtn27/envs/searchr1/lib/python3.10/site-packages/vllm/`.

## Memory Notes

**8-GPU setup** (recommended):
- ~40-50GB per GPU
- Full batch sizes, fast training

**Single-GPU** (limited):
- Validation works
- Training OOMs with default settings
- To use 1 GPU: reduce batch sizes in `train.sh`:
  ```bash
  data.train_batch_size=16
  actor_rollout_ref.actor.ppo_mini_batch_size=16
  actor_rollout_ref.rollout.n_agent=4
  ```

## Troubleshooting

**Retriever not found**: Start the server (step 3)

**OOM errors**: 
- Check you have 8 GPUs: `nvidia-smi`
- Or reduce batch sizes (see Memory Notes)

**Import errors**: Run `bash setup_environment.sh` again

## Files

```
├── environment.yml              # Conda dependencies
├── setup_environment.sh         # Install script
├── prepare_hotpotqa_data.py     # Data prep
├── retrieval_server.py          # E5 retrieval server
├── train.sh                     # Training launcher
├── data/
│   ├── train.parquet           # 7k training questions
│   ├── test.parquet            # 100 validation questions
│   └── index/e5_Flat.index     # FAISS index
└── Search-R1/                   # Modified Search-R1 repo
    └── verl/utils/reward_score/qa_f1.py  # F1 reward function
```

## Citation

```bibtex
@article{search-r1-2024,
  title={Search-R1: Reasoning with Search in Multi-Hop Question Answering},
  year={2024}
}
```
