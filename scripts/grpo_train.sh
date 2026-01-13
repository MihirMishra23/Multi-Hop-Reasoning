#!/bin/bash

# GRPO Training Script for LMLM Multi-Hop QA
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

export WANDB_PROJECT=lmlm_multi_hop
# source ./scripts/account/wandb_config.sh

# Default values
GPU_TYPE="B200"
MODEL_PATH="/share/j_sun/lz586/checkpoints/lmlm_multi_hop/qwen3-1.7B_sft_v1.3_5743"
DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_train_start_idx_82347_nb_8100_database.json"
OUTPUT_DIR="${MODEL_PATH}-GRPO"
NUM_GPUS=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu_type)
            GPU_TYPE="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --database_path)
            DATABASE_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ "$GPU_TYPE" == "B200" ]; then
    # B200
    NUM_GPUS=1
    NUM_GENERATIONS=2
    PER_DEVICE_TRAIN_BATCH_SIZE=16
    GRADIENT_ACCUMULATION_STEPS=1
    VLLM_GPU_MEMORY_UTILIZATION=0.2
elif [ "$GPU_TYPE" == "H100" ]; then
    # H100 debug
    NUM_GPUS=2
    NUM_GENERATIONS=4
    PER_DEVICE_TRAIN_BATCH_SIZE=4
    GRADIENT_ACCUMULATION_STEPS=1
    VLLM_GPU_MEMORY_UTILIZATION=0.2
fi

if [ "$NUM_GPUS" == 2 ]; then
    export CUDA_VISIBLE_DEVICES=0,1
elif [ "$NUM_GPUS" == 1 ]; then
    export CUDA_VISIBLE_DEVICES=0
else
    echo "Invalid number of GPUs: ${NUM_GPUS}"
    exit 1
fi

echo "Starting GRPO training with:"
echo "  Model: ${MODEL_PATH}"
echo "  Database: ${DATABASE_PATH}"
echo "  Output: ${OUTPUT_DIR}"
echo "  GPUs: ${NUM_GPUS}"
echo "  GPU Type: ${GPU_TYPE}"

accelerate launch \
  --num_processes=${NUM_GPUS} \
  --config_file=configs/accelerate/multi_gpu_${NUM_GPUS}.yaml \
  src/grpo_train.py \
  --model_path="${MODEL_PATH}" \
  --database_path="${DATABASE_PATH}" \
  --output_dir="${OUTPUT_DIR}" \
  --num_generations=${NUM_GENERATIONS} \
  --num_generations_eval=${NUM_GENERATIONS} \
  --per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
  --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
  --max_completion_length=256 \
  --eval_steps=4 \
  --logging_steps=1 \
  --vllm_gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION} \
  --use_vllm \
  --vllm_mode=colocate \
  --adaptive_k \
  --tools \
  --gradient_checkpointing \
  --do_eval \
  --log_completions \
  --beta=0.001 \
  --learning_rate=5e-7 \
  --loss_type=grpo \
  --max_grad_norm=1.0 \
  --warmup_ratio=0.1 \
  --top_k=0

echo "Training completed!"