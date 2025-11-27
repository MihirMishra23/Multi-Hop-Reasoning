#!/bin/bash

export WANDB_PROJECT=tofu_ft
export WANB_NAME=qwen3-1.7B-nov18
source ./scripts/account/wandb_config.sh

# Unique port per job
export MASTER_PORT=$((29501 + RANDOM % 1000))

export NCCL_TIMEOUT=18000  # 5 hours in seconds
export NCCL_ASYNC_ERROR_HANDLING=1


# Prevent NCCL conflicts
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export TORCH_USE_CUDA_DSA=1



OUTPUT_ROOT="training/qwen3-1.7b/nov18/checkpoints"

# MODEL=llama-1b-warmup
# MODEL=LMLM-M
# MODEL=llama-1b
# MODEL=LMLM-M

if [ "$MODEL" = "LMLM-S" ]; then
    MODEL_NAME_OR_PATH="tiny-llama2-176M"
    CKPT_PATH=kilian-group/LMLM-llama2-176M
elif [ "$MODEL" = "LMLM-M" ]; then
    MODEL_NAME_OR_PATH="tiny-llama2-382M"
    CKPT_PATH=kilian-group/LMLM-llama2-382M
elif [ "$MODEL" = "llama-1b" ]; then
    MODEL_NAME_OR_PATH="meta-llama/Llama-3.2-1B-Instruct"
    CKPT_PATH=${MODEL_NAME_OR_PATH}
elif [ "$MODEL" = "llama-1b-warmup" ]; then
    MODEL_NAME_OR_PATH=${MODEL}
    CKPT_PATH="/share/j_sun/lz586/checkpoints/tofu_ft_v7/Llama-3.2-1B-Instruct_warmup/checkpoint-657"
fi

MODEL_NAME_OR_PATH="Qwen/Qwen3-1.7B"
CKPT_PATH="yes"


NUM_TRAIN_EPOCHS=10
NUM_GPUs=1
PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4
EVAL_ACCUMULATION_STEPS=1 #number of steps before copying metrics to CPU, avoids OOM


ADD_DBLOOKUP_TOKENS=True

# Compute effective batch size
EFFECTIVE_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUs))


DATASPLITE_LST=("full")
# DATASPLITE_LST=("full" "retain")
for DATASPLITE in "${DATASPLITE_LST[@]}"; do

        
    if [ "$DATASPLITE" = "full" ]; then
        DATASET_PATH=./data/lmlm_trajectories_1k_converted.json
    elif [ "$DATASPLITE" = "retain" ]; then
        DATASET_PATH=../unlearning/open-unlearning/data/annotation/tofu-train-retain3.8k_chatgpt_gpt4o-v7.1_qa.json
    fi

    # Add "new" if ADD_DBLOOKUP_TOKENS=True
    if [ "$ADD_DBLOOKUP_TOKENS" = "True" ]; then
        OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL}_${DATASPLITE}_ep${NUM_TRAIN_EPOCHS}_bsz${EFFECTIVE_BATCH_SIZE}_new_qa"
    else
        OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL}_${DATASPLITE}_ep${NUM_TRAIN_EPOCHS}_bsz${EFFECTIVE_BATCH_SIZE}_qa"
    fi
    
    echo "Running for $DATASPLITE with dataset: $DATASET_PATH"
    echo "Output directory: $OUTPUT_DIR"


python \
    finetune.py \
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
    --bf16 \
    --bf16_full_eval \
    --weight_decay 0.01 \
    --optim paged_adamw_32bit \
    --lr_scheduler_type cosine \
    --do_train \
    --eval_strategy epoch \
    --eval_steps 100 \
    --save_strategy epoch \
    --save_steps 100 \
    --save_total_limit 3 \
    --save_only_model \
    --logging_steps 10 \
    --logging_dir ${OUTPUT_DIR}/logs \
    --ddp_find_unused_parameters true \
    --warmup_ratio 0.2 \
    --resume_from_checkpoint ${CKPT_PATH} \
    --eval_accumulation_steps ${EVAL_ACCUMULATION_STEPS} \
    --max_seq_length 1024 
    # --gradient_checkpointing \
    # --eval_on_start
    # --do_eval \
    # --truncation \
    # --padding "max_length" \
    # --eval_on_start \
done