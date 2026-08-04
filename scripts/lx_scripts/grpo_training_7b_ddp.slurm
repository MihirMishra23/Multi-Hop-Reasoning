#!/bin/bash


# 7B GRPO in DDP mode (FSDP+vLLM-colocate weight sync confirmed broken via probe).
# 4xB200 OOMed even at bsz=2 (~176 GiB used, no margin) — DDP forces full
# AdamW+grad+master copy replication on every rank. Bumping to 8xB200:
#   * vLLM TP=8 shrinks per-GPU vLLM weights to ~1.75 GiB
#   * vllm_util=0.10 leaves plenty of KV headroom (TP=8 KV per prompt is tiny)
#   * Training state still replicated per-rank (DDP), but with more total memory
#     and the 7B branch's default per_device=4 stands.
#
#SBATCH -J grpo_qwen2.5_7b_ddp
#SBATCH -o /home/lz586/icl/slurm_output/lmlm_multihop/grpo_qwen2.5_7b_ddp_%j.out
#SBATCH -e /home/lz586/icl/slurm_output/lmlm_multihop/grpo_qwen2.5_7b_ddp_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH -t infinite
#SBATCH --partition=aimi
#SBATCH --gres=gpu:nvidia_b200:8
#SBATCH --requeue


# Load the required environment
source activate mem

# Navigate to the working directory
cd /home/lz586/icl/Multi-Hop-Reasoning/


DATASET=hotpotqa-v2

# Same checkpoint-368 the FSDP run used, so the only changed variable is DDP vs FSDP.
MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen2.5-7B-SFT_hotpotqa_ep3_bsz48_lr5e-5_th-1/checkpoint-368

DATABASE_PATH=""
THRESHOLD=-1
REWARD_FUNC="f1"
TWO_PHASE=""
DEBUG=""

export TRITON_CACHE_DIR=/home/lz586/.cache/triton
export TMPDIR=/home/lz586/.tmp
mkdir -p $TRITON_CACHE_DIR $TMPDIR

# Help PyTorch reclaim fragmented allocator slabs — saves ~1-2 GiB
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Overrides ────────────────────────────────────────────────────────────────
# FORCE_DDP: use DDP (multi_gpu) instead of the 7B branch's default FSDP.
# 8xB200 strategy: keep 7B branch's per_device=4 default (grad_accum=16 with 8 GPUs).
# vllm_util=0.10 — KV cache requirement per GPU is tiny under TP=8, so we'd rather
# give the surplus to PyTorch's allocator.
export FORCE_DDP=1
export VLLM_UTIL_OVERRIDE=0.10

if [ "${DATASET}" = "hotpotqa-v2" ]; then
    echo "Training on hotpotqa-v2"
    TRAIN_SIZE=7000
    TOTAL_BATCH_SIZE=512
    DATASET_NAME="hotpotqa"
    TWO_PHASE="--two_phase"
elif [ "${DATASET}" = "hotpotqa-v2-onlyphase1" ]; then
    TRAIN_SIZE=7000
    TOTAL_BATCH_SIZE=512
    DATASET_NAME="hotpotqa"
    TWO_PHASE=""
    DATABASE_PATH=/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_train_start_idx_82347_nb_8100_database.json
elif [ "${DATASET}" = "hotpotqa-v2-debug" ]; then
    TRAIN_SIZE=1000
    TOTAL_BATCH_SIZE=256
    DATASET_NAME="hotpotqa"
    DEBUG="--debug"
    TWO_PHASE="--two_phase"
else
    echo "Invalid dataset: ${DATASET}"; exit 1
fi

USE_CHAT_TEMPLATE=""
if [[ "${MODEL_PATH}" == *"Qwen/Qwen"* ]]; then
    PHASE1_PROMPT_TYPE="formatted_zero_rl_v5"
    USE_CHAT_TEMPLATE="--use_chat_template"
else
    PHASE1_PROMPT_TYPE="sft"
fi

echo "Starting GRPO Training (Qwen2.5-7B ablation, 8xB200 DDP)..."
bash /home/lz586/icl/Multi-Hop-Reasoning/scripts/grpo_train.sh \
--gpu_type B200 \
--model_path ${MODEL_PATH} \
--dataset_name ${DATASET_NAME} \
--database_path "${DATABASE_PATH}" \
--train_size ${TRAIN_SIZE} \
--reward_func ${REWARD_FUNC} \
--total_batch_size ${TOTAL_BATCH_SIZE} \
--phase1_prompt_type ${PHASE1_PROMPT_TYPE} \
${USE_CHAT_TEMPLATE} \
${TWO_PHASE} \
${DEBUG}

echo "GRPO Training complete!"
