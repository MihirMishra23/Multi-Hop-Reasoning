#!/bin/bash

# GRPO Training Script for LMLM Multi-Hop QA
#export CUDA_LAUNCH_BLOCKING=1  # Removed: causes vLLM CUDA errors to propagate synchronously to NCCL → hang at DDP init
export VLLM_BATCH_INVARIANT=0
#export TORCH_USE_CUDA_DSA=1  # Removed: paired with CUDA_LAUNCH_BLOCKING for debugging only
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# export WANDB_ENTITY=dongyoung-go-cornell-university
export WANDB_ENTITY=ryan-noonan-cornell-university
export WANDB_PROJECT=LMLM-Multihop

# DEBUG
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export VLLM_USE_FLASHINFER=0
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}


# bash /home/lz586/icl/Multi-Hop-Reasoning/scripts/grpo_train.sh --gpu_type H100
# Default values
GPU_TYPE="None"
# MODEL_PATH="Qwen/Qwen3-1.7B"
MODEL_PATH=/share/j_sun/rtn27/checkpoints/lmlm_multi_hop//Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed
#DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_train_start_idx_82347_nb_8100_database.json"
DATABASE_PATH="" # -> Not used for two phase
SAVE_DIR=/share/j_sun/lmlm_multihop/checkpoints/debug
DATASET_NAME="hotpotqa"
NUM_GPUS=1
SAVE_VERSION="full-overfit" #Put anything here, it is added to the model path

# config
LOSS_TYPE="grpo"
VLLM_GPU_MEMORY_UTILIZATION=0.15
BETA=0.0
LEARNING_RATE=5e-6
# two_phase generation dimensions (set GPU-type block below overrides NUM_GENERATIONS):
#   B  = (PER_DEVICE_TRAIN_BATCH_SIZE * NUM_GPUS) / NUM_GENERATIONS  — unique questions per step (e.g. (16*2)/8 = 4)
#   K  = NUM_DB_ROLLOUTS   — DB rollouts per question   (Phase 1 completions: B*K)
#   N  = NUM_GENERATIONS   — QA rollouts per question   (Phase 2 completions: B*N, must be divisible by K)
#   M  = N / K             — QA rollouts per (question, DB) pair
NUM_GENERATIONS=16    # N
NUM_DB_ROLLOUTS=4    # K  (set >1 to compare multiple DBs per question; N must be divisible by K)
NUM_TRAIN_EPOCHS=5 # default 3
TRAIN_SIZE=7000
EVAL_SIZE=100
MAX_COMPLETION_LENGTH=1024
# EVAL_STEPS=1
EVAL_STEPS=5000 # disable it for now
LOGGING_STEPS=5
TOP_P=0.95
TEMPERATURE=1.3
TOP_K=4
IS_ADAPTIVE_K=False
RETRIEVAL_THRESHOLD=0.6
REWARD_FUNC="em_size"
PHASE1_REWARD_TYPE="binary"
PHASE1_PROMPT_TYPE="context_only"
PHASE1_DB_WEIGHT_MODE="count_dynamic"  # none | fixed[_<w>] | dynamic | count | count_dynamic

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
        --num_train_epochs)
            NUM_TRAIN_EPOCHS="$2"
            shift 2
            ;;
        --is_adaptive_k)
            IS_ADAPTIVE_K="$2"
            shift 2
            ;;
        --two_phase)
            TWO_PHASE=1
            shift 1
            ;;
        --train_size)
            TRAIN_SIZE="$2"
            shift 2
            ;;
        --debug)
            DEBUG=1
            shift 1
            ;;
        --reward_func)
            REWARD_FUNC="$2"
            shift 2
            ;;
        --phase1_reward_type)
            PHASE1_REWARD_TYPE="$2"
            shift 2
            ;;
        --phase1_prompt_type)
            PHASE1_PROMPT_TYPE="$2"
            shift 2
            ;;
        --num_db_rollouts)
            NUM_DB_ROLLOUTS="$2"
            shift 2
            ;;
        --phase1_db_weight_mode)
            PHASE1_DB_WEIGHT_MODE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --retrieval-threshold)
            RETRIEVAL_THRESHOLD="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
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
    if [[ "${MODEL_PATH}" == *"1.7B"* ]]; then
        NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        echo "  Detected ${NUM_GPUS} GPU(s)"
        NUM_GENERATIONS=16
        GRADIENT_ACCUMULATION_STEPS=8
        PER_DEVICE_TRAIN_BATCH_SIZE=16
        VLLM_GPU_MEMORY_UTILIZATION=0.15
        # NUM_GPUS=2
        # NUM_GENERATIONS=8
        # PER_DEVICE_TRAIN_BATCH_SIZE=16
        # GRADIENT_ACCUMULATION_STEPS=8
        # VLLM_GPU_MEMORY_UTILIZATION=0.15
    elif [[ "${MODEL_PATH}" == *"4B"* ]]; then
        NUM_GPUS=2
        NUM_GENERATIONS=8
        PER_DEVICE_TRAIN_BATCH_SIZE=16
        GRADIENT_ACCUMULATION_STEPS=8
        VLLM_GPU_MEMORY_UTILIZATION=0.15
    elif [[ "${MODEL_PATH}" == *"382M"* ]]; then
        NUM_GPUS=1
        NUM_GENERATIONS=8
        PER_DEVICE_TRAIN_BATCH_SIZE=256
        GRADIENT_ACCUMULATION_STEPS=1
        VLLM_GPU_MEMORY_UTILIZATION=0.15
    else
        echo "Invalid model path: ${MODEL_PATH}"
        exit 1
    fi
elif [ "$GPU_TYPE" == "H100" ]; then
    # H100 debug
    NUM_GPUS=2
    NUM_GENERATIONS=8
    PER_DEVICE_TRAIN_BATCH_SIZE=8
    GRADIENT_ACCUMULATION_STEPS=16
    VLLM_GPU_MEMORY_UTILIZATION=0.2
fi

if [ -n "${DEBUG}" ]; then
    echo "Debug mode enabled"
    # TRAIN_SIZE=100
    # EVAL_SIZE=10
    # NUM_TRAIN_EPOCHS=10
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    RETRIEVAL_THRESHOLD=0.6
    TOP_K=4
    echo "  Detected ${NUM_GPUS} GPU(s)"

    # NUM_GENERATIONS=2
    # GRADIENT_ACCUMULATION_STEPS=8
    # PER_DEVICE_TRAIN_BATCH_SIZE=2
    # VLLM_GPU_MEMORY_UTILIZATION=0.15
fi

CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
export CUDA_VISIBLE_DEVICES
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# TODO (multi-GPU): In two_phase mode, gather(rewards_per_func) produces ordering
# [GPU0_phase1+phase2, GPU1_phase1+phase2, ...] but the advantage computation in
# lmlm_basetrainer.py assumes [all_phase1 (B*K), all_phase2 (B*N)]. This causes
# completely wrong Phase-1/Phase-2 advantage grouping when NUM_GPUS > 1.
# Fix: reorder the gathered tensor before the advantage split, or gather phase1/phase2 separately.
if [ "${TWO_PHASE}" = "--two_phase" ] && [ "${NUM_GPUS}" -gt 1 ]; then
    echo "WARNING: two_phase mode with NUM_GPUS=${NUM_GPUS} > 1 is not yet supported." >&2
    echo "         Advantage computation will be incorrect. See TODO in grpo_train.sh and lmlm_basetrainer.py." >&2
fi
# NUM_TRAIN_EPOCHS=100


#-------------------------------- Save dir --------------------------------
# Dimensions: B=unique questions/step, K=DB rollouts, M=QA rollouts per (B,K), N=K*M=num_generations
# bs=per_device_train_batch_size, s=gradient_accumulation_steps
M=$((NUM_GENERATIONS / NUM_DB_ROLLOUTS))
B=$((PER_DEVICE_TRAIN_BATCH_SIZE * NUM_GPUS / NUM_GENERATIONS))
OUTPUT_DIR="${SAVE_DIR}/${MODEL_PATH##*/}-${LOSS_TYPE}-B${B}-K${NUM_DB_ROLLOUTS}-M${M}-bs${PER_DEVICE_TRAIN_BATCH_SIZE}-s${GRADIENT_ACCUMULATION_STEPS}-b${BETA}-ep${NUM_TRAIN_EPOCHS}-n${TRAIN_SIZE}-${REWARD_FUNC}"
if [ -n "${TWO_PHASE}" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}-v2-p1${PHASE1_REWARD_TYPE}-pt${PHASE1_PROMPT_TYPE}-wm${PHASE1_DB_WEIGHT_MODE}"
    TWO_PHASE="--two_phase"
else
    TWO_PHASE=""
fi
OUTPUT_DIR="${OUTPUT_DIR}-th${RETRIEVAL_THRESHOLD}-topk${TOP_K}"
if [ -n "${DEBUG}" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}-debug"
fi

# Use it in training
LAST_CKPT=$(ls -d $OUTPUT_DIR/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
if [ -n "$LAST_CKPT" ]; then
    RESUME_FROM_CHECKPOINT="--resume_from_checkpoint=${LAST_CKPT}"
else
    RESUME_FROM_CHECKPOINT=""
fi

# "${MODEL_NAME_OR_PATH##*/}-SFT_ep${NUM_TRAIN_EPOCHS}_bsz${EFFECTIVE_BATCH_SIZE}_th${THRESHOLD}"

# split the threshold by split the _th from the model path
BASENAME="${MODEL_PATH##*/}"
THRESHOLD="${BASENAME##*_th}"

if [ "${THRESHOLD}" = "-3" ]; then
    RETURN_TRIPLES="--return_triples"
    echo "RETURN_TRIPLES: ${RETURN_TRIPLES}"
else
    RETURN_TRIPLES=""
fi

if [ "${IS_ADAPTIVE_K}" = "True" ]; then
    ADAPTIVE_K="--adaptive_k"
    echo "ADAPTIVE_K: ${ADAPTIVE_K}"
else
    ADAPTIVE_K=""
    OUTPUT_DIR="${OUTPUT_DIR}-nak"
fi

if [ "${USE_INVERSES}" = 'True' ]; then
    USE_INVERSES="--use-inverses"
else
    USE_INVERSES=""
fi


echo "Starting GRPO training with:"
echo "  Model: ${MODEL_PATH}"
echo "  Database: ${DATABASE_PATH}"
echo "  Output: ${OUTPUT_DIR}"
echo "  GPUs: ${NUM_GPUS}"
echo "  GPU Type: ${GPU_TYPE}"
echo "  Resume from checkpoint: ${RESUME_FROM_CHECKPOINT}"
echo "  Two phase: ${TWO_PHASE}"
echo "  Return triples: ${RETURN_TRIPLES}"
echo "  Adaptive k: ${ADAPTIVE_K}"


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
  --tools \
  --gradient_checkpointing \
  --do_eval \
  --log_completions \
  --beta=${BETA} \
  --learning_rate=${LEARNING_RATE} \
  --loss_type=${LOSS_TYPE} \
  --max_grad_norm=1.0 \
  --warmup_ratio=0.1 \
  --lr_scheduler_type=cosine \
  --vllm_max_model_length=4096 \
  --train_size=${TRAIN_SIZE} \
  --eval_size=${EVAL_SIZE} \
  --top_p=${TOP_P} \
  --temperature=${TEMPERATURE} \
  --top_k=${TOP_K} \
  --num_train_epochs=${NUM_TRAIN_EPOCHS} \
  --save_strategy=steps \
  --save_total_limit=5 \
  --save_steps=0.1 \
  ${RESUME_FROM_CHECKPOINT} \
  --use-inverses \
  --retrieval-threshold ${RETRIEVAL_THRESHOLD} \
  ${TWO_PHASE} \
  ${RETURN_TRIPLES} \
  ${USE_INVERSES} \
  ${ADAPTIVE_K} \
  --reward_func=${REWARD_FUNC} \
  --phase1_reward_type=${PHASE1_REWARD_TYPE} \
  --phase1_prompt_type=${PHASE1_PROMPT_TYPE} \
  --num_db_rollouts=${NUM_DB_ROLLOUTS} \
  --phase1_db_weight_mode=${PHASE1_DB_WEIGHT_MODE}

echo "Training completed!"