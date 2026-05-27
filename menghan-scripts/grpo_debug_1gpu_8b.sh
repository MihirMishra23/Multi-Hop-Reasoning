#!/bin/bash
# 1-GPU 8B GRPO smoke test (bypasses scripts/grpo_train.sh's 8B preset that requires fsdp_4.yaml).
# Goal: verify the pipeline works on 1 B200 with all the recent fixes.
#
# Memory budget (B200 180GB):
#   weight 16 + AdamW 64 + grad 16 + activations 10 + vLLM 16 + vLLM workspace 27 = ~149 GB
#   tight but fits. If OOM, lower vllm_gpu_memory_utilization to 0.10.

set -e
cd /share/j_sun/mx253/Multi-Hop-Reasoning

if [[ "$CONDA_DEFAULT_ENV" != "lmlm_b200" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate lmlm_b200
fi

# Env (same as scripts/grpo_train.sh)
export VLLM_USE_V1=0
export VLLM_USE_FLASHINFER=0
export VLLM_TORCH_COMPILE_LEVEL=0
export VLLM_BATCH_INVARIANT=0
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}
export WANDB_NAME="debug-1gpu-8b-$(date +%m%d-%H%M)"
echo "WANDB_NAME=${WANDB_NAME}"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [[ "${NUM_GPUS}" -ne 1 ]]; then
    echo "WARN: detected ${NUM_GPUS} GPUs, this script is for 1 GPU."
fi
echo "Using ${NUM_GPUS} GPU"

MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-8B-SFT_hotpotqa_ep3_bsz48_lr5e-5_th-1
SAVE_DIR=/share/j_sun/lmlm_multihop/checkpoints/debug
OUTPUT_DIR="${SAVE_DIR}/${MODEL_PATH##*/}-1gpu-debug"

# Use plain single-GPU accelerate config (no FSDP — fsdp_1.yaml doesn't exist anyway,
# and FSDP with 1 rank is a no-op).
ACCEL_CONFIG="configs/accelerate/multi_gpu_1.yaml"

# Tiny per-device batch, high grad accum to keep TBS reasonable.
# Smaller TOTAL_BATCH_SIZE (128 instead of 512) to fit memory.
PER_DEVICE_TRAIN_BATCH_SIZE=1
NUM_GENERATIONS=8
NUM_DB_ROLLOUTS=4
TOTAL_BATCH_SIZE=64
# GRADIENT_ACCUMULATION_STEPS = TBS / (PER_DEVICE * NUM_GPUS)
GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BATCH_SIZE / PER_DEVICE_TRAIN_BATCH_SIZE / NUM_GPUS))
echo "  TBS=${TOTAL_BATCH_SIZE}  PER_DEVICE=${PER_DEVICE_TRAIN_BATCH_SIZE}  GRAD_ACCUM=${GRADIENT_ACCUMULATION_STEPS}"

accelerate launch \
    --num_processes=${NUM_GPUS} \
    --config_file=${ACCEL_CONFIG} \
    src/grpo_train.py \
    --model_path="${MODEL_PATH}" \
    --database_path="" \
    --dataset_name=hotpotqa \
    --output_dir="${OUTPUT_DIR}" \
    --train_size=1000 \
    --eval_size=50 \
    --max_steps=20 \
    --num_generations=${NUM_GENERATIONS} \
    --num_generations_eval=${NUM_GENERATIONS} \
    --num_db_rollouts=${NUM_DB_ROLLOUTS} \
    --per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size=${NUM_GENERATIONS} \
    --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    --max_completion_length=1024 \
    --vllm_max_model_length=4096 \
    --vllm_gpu_memory_utilization=0.15 \
    --use_vllm \
    --vllm_mode=colocate \
    --tools \
    --gradient_checkpointing \
    --two_phase \
    --reward_func=f1 \
    --phase1_reward_type=binary \
    --phase1_prompt_type=sft \
    --phase1_db_weight_mode=count \
    --beta=0.0 \
    --learning_rate=5e-6 \
    --loss_type=grpo \
    --max_grad_norm=1.0 \
    --warmup_ratio=0.1 \
    --lr_scheduler_type=cosine \
    --temperature=1.0 \
    --top_p=0.95 \
    --top_k=4 \
    --retrieval-threshold 0.6 \
    --do_eval \
    --eval_strategy='no' \
    --save_strategy='no' \
    --logging_steps=1 \
    --report_to=wandb