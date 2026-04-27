#!/bin/bash

# ── Environment ───────────────────────────────────────────────────────────────
export WANDB_ENTITY=ryan-noonan-cornell-university
export WANDB_PROJECT=LMLM-Multihop-SFT
export MASTER_PORT=$((29501 + RANDOM % 1000))
export NCCL_TIMEOUT=18000          # 5 hours
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_USE_CUDA_DSA=1

# ── Defaults ──────────────────────────────────────────────────────────────────
OUTPUT_ROOT=/share/j_sun/lz586/checkpoints/lmlm_multi_hop
THRESHOLD=-1
DATASET=hotpotqa
TWO_PHASE=True
MAX_SEQ_LENGTH=""  # overrides per-model default when set

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_size)   MODEL_SIZE="$2";   shift 2 ;;
        --dataset_path) DATASET_PATH="$2"; shift 2 ;;
        --threshold)    THRESHOLD="$2";    shift 2 ;;
        --dataset)      DATASET="$2";      shift 2 ;;
        --two_phase)       TWO_PHASE="$2";       shift 2 ;;
        --max_seq_length)  MAX_SEQ_LENGTH="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Model config ──────────────────────────────────────────────────────────────
NUM_GPUS=1
NUM_TRAIN_EPOCHS=3
LEARNING_RATE=5e-5

case "${MODEL_SIZE}" in
    1.7B)
        MODEL_NAME_OR_PATH="Qwen/Qwen3-1.7B"
        PER_DEVICE_TRAIN_BATCH_SIZE=24
        GRADIENT_ACCUMULATION_STEPS=2
        MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
        ;;
    4B)
        MODEL_NAME_OR_PATH="Qwen/Qwen3-4B"
        PER_DEVICE_TRAIN_BATCH_SIZE=8
        GRADIENT_ACCUMULATION_STEPS=6
        MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
        ;;
    8B)
        MODEL_NAME_OR_PATH="Qwen/Qwen3-8B"
        PER_DEVICE_TRAIN_BATCH_SIZE=4
        GRADIENT_ACCUMULATION_STEPS=6
        MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
        NUM_GPUS=2
        ;;
    382M)
        MODEL_NAME_OR_PATH="kilian-group/LMLM-llama2-382M"
        PER_DEVICE_TRAIN_BATCH_SIZE=48
        GRADIENT_ACCUMULATION_STEPS=1
        MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
        NUM_TRAIN_EPOCHS=10
        ;;
    *)
        echo "Invalid model size: ${MODEL_SIZE}"; exit 1 ;;
esac

# ── Dataset path ──────────────────────────────────────────────────────────────
_SFT=/share/j_sun/lmlm_multihop/sft_data
_SFT_LZ=/share/j_sun/lz586/lmlm_multihop/sft_data

case "${DATASET}" in
    hotpotqa)
        case "${THRESHOLD}" in
            -1)
                if [ "${TWO_PHASE}" = "True" ]; then
                    DATASET_PATH="${_SFT}/gemini_2phase_rollouts_hotpotqa_6k_db_train_end_context_fifths_1203_ex_6k_qa_hotpot_rollouts_classic_retrieval_train_from_start.json"
                else
                    DATASET_PATH="${_SFT}/gemini_rollouts_hotpotqa_12k_filtered_5743_classic_retrieval.json"
                fi ;;
            0.3)  DATASET_PATH="${_SFT}/combined_01_15/combined_2026-01-15_filtered_thresh_0.3_len_13796.json" ;;
            0.8)  DATASET_PATH="${_SFT}/combined_01_15/combined_2026-01-15_filtered_thresh_0.8_len_12004.json" ;;
            1.0)  DATASET_PATH="${_SFT}/combined_01_15/combined_2026-01-15_filtered_thresh_1.0_len_11308.json" ;;
            -2)   DATASET_PATH="${_SFT}/combined_01_15/combined_2026-01-15_filtered_thresh_-2_len_18351.json" ;;
            -3)   DATASET_PATH="${_SFT}/12-19_rollouts_combined_5743_12k_return_triplets.json" ;;  # returns triplets
            *)    echo "Invalid threshold for hotpotqa: ${THRESHOLD}"; exit 1 ;;
        esac ;;
    # mquake variants all use a fixed dataset path (threshold ignored)
    mquake)             DATASET_PATH="${_SFT}/mquake-remastered_rollouts_train_th_1.0_len_5258.json" ;;  # NOTE: train/test overlap bug
    mquake-real)        DATASET_PATH="${_SFT_LZ}/REAL_q3000.json" ;;
    mquake-real-debug)  DATASET_PATH="${_SFT_LZ}/REAL_q3000_debug.json" ;;
    mquake-real5334-r1) DATASET_PATH="${_SFT_LZ}/REAL-q6334-r1_REAL_q5334_n5334.json" ;;
    mquake-cf)          DATASET_PATH="${_SFT_LZ}/CF_q300_r10.json" ;;
    mquake-cf-debug)    DATASET_PATH="${_SFT_LZ}/CF_q300_r10_debug.json" ;;
    mquake-cf5334)      DATASET_PATH="${_SFT_LZ}/CF-q6334-r10_CF_q700_r10.json" ;;
    mquake-cf5334-r1)   DATASET_PATH="${_SFT_LZ}/CF-q6334-r10_CF_q5334_r1_n5334.json" ;;
    mquake-cf5334-r10)  DATASET_PATH="${_SFT_LZ}/CF-q6334-r10_CF_q5334_r10_n5334.json" ;;
    mquake-cfall-r10)   DATASET_PATH="${_SFT_LZ}/CF-q6334-r10_CF_q5334_r10_n48602.json" ;;
    2wiki) echo "2wiki not implemented"; exit 1 ;;
    *)     echo "Invalid dataset: ${DATASET}"; exit 1 ;;
esac

# ── Output / run naming ───────────────────────────────────────────────────────
EFFECTIVE_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))
export WANDB_NAME="${MODEL_NAME_OR_PATH##*/}-SFT_${DATASET}_ep${NUM_TRAIN_EPOCHS}_bsz${EFFECTIVE_BATCH_SIZE}_lr${LEARNING_RATE}_th${THRESHOLD}"
[[ "${TWO_PHASE}" == "False" ]] && export WANDB_NAME="${WANDB_NAME}_1phase"
OUTPUT_DIR="${OUTPUT_ROOT}/${WANDB_NAME}"

echo "Dataset:    ${DATASET_PATH}"
echo "Output dir: ${OUTPUT_DIR}"

# ── Launch ────────────────────────────────────────────────────────────────────
accelerate launch \
  --num_processes=${NUM_GPUS} \
  --config_file=configs/accelerate/multi_gpu_${NUM_GPUS}.yaml \
    src/sft_train.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --dataset_name ${DATASET_PATH} \
    --dataset_text_field None \
    --output_dir ${OUTPUT_DIR} \
    --use_special_dblookup_tokens True \
    --plain_baseline False \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --do_train \
    --eval_strategy epoch \
    --save_strategy steps \
    --save_steps 0.125 \
    --save_total_limit 8 \
    --save_only_model \
    --logging_steps 10 \
    --logging_dir ${OUTPUT_DIR}/logs \
    --warmup_ratio 0.1 \
    --eval_accumulation_steps 1 \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --bf16 True \
    --resume_from_checkpoint ${MODEL_NAME_OR_PATH} \
    --gradient_checkpointing
    # --do_eval \
    # --eval_on_start \
    # --truncation \
    # --padding "max_length" \
