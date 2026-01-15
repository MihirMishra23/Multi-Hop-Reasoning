#!/bin/bash

# GRPO Training Script for LMLM Multi-Hop QA
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WANDB_PROJECT=lmlm_multi_hop

# DEBUG
export VLLM_ATTENTION_BACKEND=FLASH_ATTN   # or TRITON
export VLLM_USE_FLASHINFER=0


# bash /home/lz586/icl/Multi-Hop-Reasoning/scripts/grpo_train.sh --gpu_type B200
# Default values
GPU_TYPE="B200"
# MODEL_PATH="Qwen/Qwen3-1.7B"
MODEL_PATH="/share/j_sun/lz586/checkpoints/lmlm_multi_hop/qwen3-1.7B_sft_v1.3_5743"
DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_train_start_idx_82347_nb_8100_database.json"
SAVE_DIR=/share/j_sun/lz586/checkpoints/lmlm_multi_hop
DATASET_NAME="hotpotqa/hotpot_qa"
NUM_GPUS=1

# config
LOSS_TYPE="grpo"
VLLM_GPU_MEMORY_UTILIZATION=0.15
BETA=0.0
LEARNING_RATE=1e-6
GRADIENT_ACCUMULATION_STEPS=4
PER_DEVICE_TRAIN_BATCH_SIZE=16
NUM_GENERATIONS=8
NUM_TRAIN_EPOCHS=5 # default 3
TRAIN_SIZE=8000
EVAL_SIZE=100
MAX_COMPLETION_LENGTH=1024
EVAL_STEPS=4
LOGGING_STEPS=5
TOP_P=0.95
TEMPERATURE=1.3
TOP_K=0



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
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --dataset_name)
            DATASET_NAME="$2"
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
    NUM_GPUS=2
    NUM_GENERATIONS=8
    PER_DEVICE_TRAIN_BATCH_SIZE=16
    GRADIENT_ACCUMULATION_STEPS=8
    VLLM_GPU_MEMORY_UTILIZATION=0.15
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
# output_dir = script_args.model_path.split('/')[-1]+'-'+str(grpo_config.loss_type)+'-g'+str(grpo_config.num_generations)+'-bs'+str(grpo_config.per_device_train_batch_size)+'-s'+str(grpo_config.gradient_accumulation_steps)+'-b'+str(grpo_config.beta)+'-ep'+str(grpo_config.num_train_epochs)+'-n'+str(script_args.train_size)
OUTPUT_DIR="${SAVE_DIR}/${MODEL_PATH##*/}-${LOSS_TYPE}-g${NUM_GENERATIONS}-bs${PER_DEVICE_TRAIN_BATCH_SIZE}-s${GRADIENT_ACCUMULATION_STEPS}-b${BETA}-ep${NUM_TRAIN_EPOCHS}-n${TRAIN_SIZE}"

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
  --dataset_name="${DATASET_NAME}" \
  --database_path="${DATABASE_PATH}" \
  --output_dir="${OUTPUT_DIR}" \
  --num_generations=${NUM_GENERATIONS} \
  --num_generations_eval=${NUM_GENERATIONS} \
  --per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
  --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
  --max_completion_length=${MAX_COMPLETION_LENGTH} \
  --eval_steps=${EVAL_STEPS} \
  --logging_steps=${LOGGING_STEPS} \
  --vllm_gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION} \
  --use_vllm \
  --vllm_mode=colocate \
  --adaptive_k \
  --tools \
  --gradient_checkpointing \
  --do_eval \
  --log_completions \
  --beta=${BETA} \
  --learning_rate=${LEARNING_RATE} \
  --loss_type=${LOSS_TYPE} \
  --max_grad_norm=1.0 \
  --warmup_ratio=0.1 \
  --vllm_max_model_length=2048 \
  --train_size=${TRAIN_SIZE} \
  --eval_size=${EVAL_SIZE} \
  --top_p=${TOP_P} \
  --temperature=${TEMPERATURE} \
  --top_k=${TOP_K} \
  --num_train_epochs=${NUM_TRAIN_EPOCHS} \
  --save_strategy=steps \
  --save_total_limit=4 \
  --save_steps=0.25

echo "Training completed!"