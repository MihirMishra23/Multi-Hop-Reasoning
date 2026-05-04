#!/bin/bash

# GRPO Training Script for LMLM Multi-Hop QA

# ── Environment ───────────────────────────────────────────────────────────────
export VLLM_USE_V1=0               # Force V0 engine; V1 has CUDA illegal memory access bug
export VLLM_USE_FLASHINFER=0
export VLLM_TORCH_COMPILE_LEVEL=0  # Disable torch.compile to avoid inductor cache dir assertion errors
export VLLM_BATCH_INVARIANT=0
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}

export WANDB_ENTITY=ryan-noonan-cornell-university
export WANDB_PROJECT=LMLM-Multihop

# ── Paths ─────────────────────────────────────────────────────────────────────
GPU_TYPE=""
MODEL_PATH=""
DATABASE_PATH=""  # Not used in two-phase mode
SAVE_DIR=/share/j_sun/lmlm_multihop/checkpoints/main
DATASET_NAME="hotpotqa"
NUM_GPUS=1

# ── Batch / generation dimensions ─────────────────────────────────────────────
# N = NUM_GENERATIONS  — total rollouts per question (phase 1 + phase 2), N must be divisible by K
# K = NUM_DB_ROLLOUTS  — phase 1 DB rollouts per question            (phase 1 completions: B*K)
# M = (N-K)/K          — phase 2 QA rollouts per (question, DB) pair (phase 2 completions: B*K*M)
# B = TBS/N            — unique questions per global batch           (total: B*K + B*K*M = B*N = TBS ✓)
# GRADIENT_ACCUMULATION_STEPS = TBS / (PER_DEVICE_TRAIN_BATCH_SIZE * NUM_GPUS)  — derived
#
# Example: TBS=1024, N=32, K=4 → B=32, M=7, total=32*(4+4*7)=1024 ✓
NUM_GENERATIONS=32            # N
NUM_DB_ROLLOUTS=4             # K  (N must be divisible by K)
TOTAL_BATCH_SIZE=1024

PER_DEVICE_TRAIN_BATCH_SIZE=16
PER_DEVICE_EVAL_BATCH_SIZE=32
VLLM_GPU_MEMORY_UTILIZATION=0.6

# ── Training hyperparameters ──────────────────────────────────────────────────
LOSS_TYPE="grpo"
BETA=0.0
LEARNING_RATE=1e-6
MAX_STEPS=500
NUM_TRAIN_EPOCHS=100          # High ceiling; MAX_STEPS takes effect first
TRAIN_SIZE=7000
EVAL_SIZE=100
MAX_COMPLETION_LENGTH=1024

# ── Sampling ──────────────────────────────────────────────────────────────────
TOP_P=0.95
TEMPERATURE=1
TOP_K=4

# ── Logging / checkpointing ───────────────────────────────────────────────────
LOGGING_STEPS=5
SAVE_STEPS=25
EVAL_STEPS=500

# ── Core LMLM flags ───────────────────────────────────────────────────────────
TOOLS="--tools"               # tools enabled by default; pass --no_tools to disable
TWO_PHASE=""
RETRIEVAL_THRESHOLD=0.6
REWARD_FUNC="em"
PHASE1_REWARD_TYPE="binary"
PHASE1_PROMPT_TYPE="sft"
PHASE1_DB_WEIGHT_MODE="count"  # none | fixed | dynamic | count | count_dynamic
USE_CHAT_TEMPLATE=""

# ── Ablation flags (off by default) ───────────────────────────────────────────
USE_ADAPTIVE_K=False
USE_INVERSES=""
VANILLA_GRPO=""
RETURN_TRIPLES=""
TIER_PATH=""
TIER_MIN_SCORE=1
TIER_MAX_SCORE=7
CURRICULUM=""
CURRICULUM_PHASES="5-7,3-7,1-7"
CURRICULUM_STEPS="0.33,0.67"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu_type)              GPU_TYPE="$2";               shift 2 ;;
        --model_path)            MODEL_PATH="$2";             shift 2 ;;
        --database_path)         DATABASE_PATH="$2";          shift 2 ;;
        --save_dir)              SAVE_DIR="$2";               shift 2 ;;
        --num_gpus)              NUM_GPUS="$2";               shift 2 ;;
        --dataset_name)          DATASET_NAME="$2";           shift 2 ;;
        --num_train_epochs)      NUM_TRAIN_EPOCHS="$2";       shift 2 ;;
        --max_steps)             MAX_STEPS="$2";              shift 2 ;;
        --train_size)            TRAIN_SIZE="$2";             shift 2 ;;
        --total_batch_size)      TOTAL_BATCH_SIZE="$2";       shift 2 ;;
        --per_device_batch_size) PER_DEVICE_TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --num_generations)       NUM_GENERATIONS="$2";        shift 2 ;;
        --num_db_rollouts)       NUM_DB_ROLLOUTS="$2";        shift 2 ;;
        --learning_rate)         LEARNING_RATE="$2";          shift 2 ;;
        --retrieval_threshold)   RETRIEVAL_THRESHOLD="$2";    shift 2 ;;
        --top_k)                 TOP_K="$2";                  shift 2 ;;
        --reward_func)           REWARD_FUNC="$2";            shift 2 ;;
        --phase1_reward_type)    PHASE1_REWARD_TYPE="$2";     shift 2 ;;
        --phase1_prompt_type)    PHASE1_PROMPT_TYPE="$2";     shift 2 ;;
        --phase1_db_weight_mode) PHASE1_DB_WEIGHT_MODE="$2";  shift 2 ;;
        # Core flags
        --two_phase)             TWO_PHASE="--two_phase";     shift 1 ;;
        --use_chat_template)     USE_CHAT_TEMPLATE="--use_chat_template"; shift 1 ;;
        --no_tools)              TOOLS="";                    shift 1 ;;
        # Ablation flags
        --use_adaptive_k)        USE_ADAPTIVE_K="$2";         shift 2 ;;
        --use_inverses)          USE_INVERSES="--use_inverses"; shift 1 ;;
        --vanilla_grpo)          VANILLA_GRPO="--vanilla_grpo"; shift 1 ;;
        --return_triples)        RETURN_TRIPLES="--return_triples"; shift 1 ;;
        --tier_path)             TIER_PATH="$2";              shift 2 ;;
        --tier_min_score)        TIER_MIN_SCORE="$2";         shift 2 ;;
        --tier_max_score)        TIER_MAX_SCORE="$2";         shift 2 ;;
        --curriculum)            CURRICULUM="--curriculum";   shift 1 ;;
        --curriculum_phases)     CURRICULUM_PHASES="$2";      shift 2 ;;
        --curriculum_steps)      CURRICULUM_STEPS="$2";       shift 2 ;;
        # Misc
        --debug)                 DEBUG=1;                     shift 1 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ── GPU-type presets ──────────────────────────────────────────────────────────
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
ACCEL_CONFIG="configs/accelerate/multi_gpu_${NUM_GPUS}.yaml"  # overridden for large models

if [ "$GPU_TYPE" == "B200" ]; then
    if [[ "${MODEL_PATH}" == *"1.7B"* ]]; then
        PER_DEVICE_TRAIN_BATCH_SIZE=16
        LEARNING_RATE=5e-6
        VLLM_GPU_MEMORY_UTILIZATION=0.4
    elif [[ "${MODEL_PATH}" == *"4B"* ]]; then
        LEARNING_RATE=5e-6
        PER_DEVICE_TRAIN_BATCH_SIZE=8
        VLLM_GPU_MEMORY_UTILIZATION=0.15
    elif [[ "${MODEL_PATH}" == *"8B"* ]]; then
        LEARNING_RATE=5e-6
        PER_DEVICE_TRAIN_BATCH_SIZE=4
        VLLM_GPU_MEMORY_UTILIZATION=0.2
        ACCEL_CONFIG="configs/accelerate/fsdp_${NUM_GPUS}.yaml"
    elif [[ "${MODEL_PATH}" == *"382M"* ]]; then
        PER_DEVICE_TRAIN_BATCH_SIZE=256
        VLLM_GPU_MEMORY_UTILIZATION=0.15
    else
        echo "Error: unsupported model size for B200 preset: ${MODEL_PATH}"
        exit 1
    fi
elif [ "$GPU_TYPE" == "H100" ]; then
    PER_DEVICE_TRAIN_BATCH_SIZE=8
    VLLM_GPU_MEMORY_UTILIZATION=0.15
fi

if [ -n "${DEBUG}" ]; then
    echo "Debug mode enabled"
    TRAIN_SIZE=1000
    EVAL_SIZE=10
fi

CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
export CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Derive GRADIENT_ACCUMULATION_STEPS from TOTAL_BATCH_SIZE
GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BATCH_SIZE / (PER_DEVICE_TRAIN_BATCH_SIZE * NUM_GPUS)))
if [ "$((GRADIENT_ACCUMULATION_STEPS * PER_DEVICE_TRAIN_BATCH_SIZE * NUM_GPUS))" -ne "${TOTAL_BATCH_SIZE}" ]; then
    echo "Error: TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE} is not divisible by PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE} * NUM_GPUS=${NUM_GPUS}" >&2
    exit 1
fi
echo "  GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS} (= ${TOTAL_BATCH_SIZE} / (${PER_DEVICE_TRAIN_BATCH_SIZE} * ${NUM_GPUS}))"

# ── Output directory ──────────────────────────────────────────────────────────
B=$((TOTAL_BATCH_SIZE / NUM_GENERATIONS))
M=$(((NUM_GENERATIONS - NUM_DB_ROLLOUTS) / NUM_DB_ROLLOUTS))

# Core name
OUTPUT_DIR="${SAVE_DIR}/${MODEL_PATH##*/}-${LOSS_TYPE}-tbs${TOTAL_BATCH_SIZE}-N${NUM_GENERATIONS}-K${NUM_DB_ROLLOUTS}-B${B}-M${M}-b${BETA}-lr${LEARNING_RATE}-step${MAX_STEPS}-n${TRAIN_SIZE}-${REWARD_FUNC}"
if [ -n "${TWO_PHASE}" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}-2ph"
    [ "${PHASE1_REWARD_TYPE}" != "binary" ] && OUTPUT_DIR="${OUTPUT_DIR}-rw${PHASE1_REWARD_TYPE}"
    OUTPUT_DIR="${OUTPUT_DIR}-pr${PHASE1_PROMPT_TYPE}-w${PHASE1_DB_WEIGHT_MODE}"
fi
OUTPUT_DIR="${OUTPUT_DIR}-th${RETRIEVAL_THRESHOLD}-topk${TOP_K}"

# Ablation suffix
[ "${USE_ADAPTIVE_K}" != "True" ] && OUTPUT_DIR="${OUTPUT_DIR}-nak"
[ -n "${TIER_PATH}" ]             && OUTPUT_DIR="${OUTPUT_DIR}-tier${TIER_MIN_SCORE}_${TIER_MAX_SCORE}"
[ -n "${CURRICULUM}" ]            && OUTPUT_DIR="${OUTPUT_DIR}-curric"
[ -n "${USE_INVERSES}" ]          && OUTPUT_DIR="${OUTPUT_DIR}-inv"
[ -n "${VANILLA_GRPO}" ]          && OUTPUT_DIR="${OUTPUT_DIR}-vanilla"
[ -n "${DEBUG}" ]                 && OUTPUT_DIR="${OUTPUT_DIR}-debug"

# ── Flag resolution ───────────────────────────────────────────────────────────
[ "${USE_ADAPTIVE_K}" = "True" ] && ADAPTIVE_K="--adaptive_k" || ADAPTIVE_K=""

# ── Resume from checkpoint ────────────────────────────────────────────────────
LAST_CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
RESUME_FROM_CHECKPOINT=${LAST_CKPT:+"--resume_from_checkpoint=${LAST_CKPT}"}

# ── Summary ───────────────────────────────────────────────────────────────────
echo "Starting GRPO training:"
echo "  Model:       ${MODEL_PATH}"
echo "  Output:      ${OUTPUT_DIR}"
echo "  GPUs:        ${NUM_GPUS} (${GPU_TYPE:-default})"
echo "  Two-phase:   ${TWO_PHASE:-off}"
echo "  Checkpoint:  ${RESUME_FROM_CHECKPOINT:-none}"

# ── Launch ────────────────────────────────────────────────────────────────────
accelerate launch \
  --num_processes=${NUM_GPUS} \
  --config_file=${ACCEL_CONFIG} \
  src/grpo_train.py \
  --model_path="${MODEL_PATH}" \
  --dataset_name="${DATASET_NAME}" \
  --database_path="${DATABASE_PATH}" \
  --output_dir="${OUTPUT_DIR}" \
  --num_generations=${NUM_GENERATIONS} \
  --num_generations_eval=${NUM_GENERATIONS} \
  --per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
  --per_device_eval_batch_size=${PER_DEVICE_EVAL_BATCH_SIZE} \
  --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
  --max_completion_length=${MAX_COMPLETION_LENGTH} \
  --logging_steps=${LOGGING_STEPS} \
  --vllm_gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION} \
  --use_vllm \
  --vllm_mode=colocate \
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
  --max_steps=${MAX_STEPS} \
  --save_strategy=steps \
  --save_total_limit=5 \
  --save_steps=${SAVE_STEPS} \
  --eval_strategy=steps \
  --eval_steps=${EVAL_STEPS} \
  --retrieval_threshold=${RETRIEVAL_THRESHOLD} \
  --reward_func=${REWARD_FUNC} \
  --phase1_reward_type=${PHASE1_REWARD_TYPE} \
  --phase1_prompt_type=${PHASE1_PROMPT_TYPE} \
  --num_db_rollouts=${NUM_DB_ROLLOUTS} \
  --phase1_db_weight_mode=${PHASE1_DB_WEIGHT_MODE} \
  --tier_min_score=${TIER_MIN_SCORE} \
  --tier_max_score=${TIER_MAX_SCORE} \
  ${TWO_PHASE} \
  ${TOOLS} \
  ${USE_CHAT_TEMPLATE} \
  ${ADAPTIVE_K} \
  ${VANILLA_GRPO} \
  ${RETURN_TRIPLES} \
  ${USE_INVERSES} \
  ${CURRICULUM} \
  ${RESUME_FROM_CHECKPOINT} \
  ${TIER_PATH:+--tier_path=${TIER_PATH}} \
  $([ -n "${CURRICULUM}" ] && echo "--curriculum_phases=${CURRICULUM_PHASES} --curriculum_steps=${CURRICULUM_STEPS}")

echo "Training completed!"