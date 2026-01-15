#!/bin/bash

export WANDB_ENTITY=ryan-noonan-cornell-university
export WANDB_PROJECT=LMLM-Multihop-SFT

# Unique port per job
export MASTER_PORT=$((29501 + RANDOM % 1000))

export NCCL_TIMEOUT=18000  # 5 hours in seconds
export NCCL_ASYNC_ERROR_HANDLING=1


# Prevent NCCL conflicts
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export TORCH_USE_CUDA_DSA=1

# input arguments for 
OUTPUT_ROOT=/share/j_sun/lz586/checkpoints/lmlm_multi_hop
DATASET_PATH=/share/j_sun/lmlm_multihop/sft_data/12-19_rollouts_combined_12k_5743_examples_6000_triplets_filtered.json
MODEL_NAME_OR_PATH=Qwen/Qwen3-1.7B

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
NUM_TRAIN_EPOCHS=5 #change to 5

if [ "${MODEL_SIZE}" = "1.7B" ]; then
    MODEL_NAME_OR_PATH="Qwen/Qwen3-1.7B"
    PER_DEVICE_TRAIN_BATCH_SIZE=24
    GRADIENT_ACCUMULATION_STEPS=2   # change to 4
    MAX_SEQ_LENGTH=2048

elif [ "${MODEL_SIZE}" = "4B" ]; then
    MODEL_NAME_OR_PATH="Qwen/Qwen3-4B"
    PER_DEVICE_TRAIN_BATCH_SIZE=8
    GRADIENT_ACCUMULATION_STEPS=6   # change to 4
    MAX_SEQ_LENGTH=1024

else
    echo "Invalid model size: ${MODEL_SIZE}"
    exit 1
fi

EVAL_ACCUMULATION_STEPS=1 #number of steps before copying metrics to CPU, avoids OOM

ADD_DBLOOKUP_TOKENS=True


# Compute effective batch size
EFFECTIVE_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))

export WANDB_NAME="${MODEL_NAME_OR_PATH##*/}-SFT_ep${NUM_TRAIN_EPOCHS}_bsz${EFFECTIVE_BATCH_SIZE}"
OUTPUT_DIR="${OUTPUT_ROOT}/${WANDB_NAME}"

echo "Running for $DATASET_PATH"
echo "Output directory: $OUTPUT_DIR"


accelerate launch \
  --num_processes=${NUM_GPUS} \
  --config_file=configs/accelerate/multi_gpu_${NUM_GPUS}.yaml \
    src/sft_train.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --dataset_name ${DATASET_PATH} \
    --dataset_text_field None \
    --output_dir ${OUTPUT_DIR} \
    --use_special_dblookup_tokens ${ADD_DBLOOKUP_TOKENS} \
    --plain_baseline False \
    --learning_rate 5e-5 \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --do_train \
    --eval_strategy epoch \
    --save_strategy steps \
    --save_steps 0.5 \
    --save_total_limit 2 \
    --save_only_model \
    --logging_steps 10 \
    --logging_dir ${OUTPUT_DIR}/logs \
    --warmup_ratio 0.1 \
    --eval_accumulation_steps ${EVAL_ACCUMULATION_STEPS} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --bf16 True \
    --resume_from_checkpoint ${MODEL_NAME_OR_PATH} \
    --gradient_checkpointing \
    # --eval_on_start
    # --do_eval \
    # --truncation \
    # --padding "max_length" \
    # --eval_on_start \
done