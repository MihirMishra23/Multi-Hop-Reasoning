#!/bin/bash

# GRPO launcher for the mx253/atlas_rag collaboration run.
#
# This is a customized copy of /share/j_sun/mx253/Multi-Hop-Reasoning/scripts/grpo_train.sh
# kept in lz586's tree so we can tune knobs without touching mx253's repo.
# It still invokes `src/grpo_train.py` and `configs/accelerate/...` relative to
# the cwd, so the caller MUST `cd /share/j_sun/mx253/Multi-Hop-Reasoning` first.
#
# Diffs vs mx253's script:
#   - H100 preset bumped from vllm_gpu_memory_utilization=0.15 to 0.22
#     (rank 0 hot due to ref-policy replica; observed ~80% at 0.15 leaves
#      ~14 GB headroom on rank 0, ~32 GB on ranks 1-3).
#   - New CLI flag --vllm_gpu_memory_utilization for ad-hoc overrides (applied
#     AFTER the GPU_TYPE preset so it actually sticks).

# ── Environment ───────────────────────────────────────────────────────────────
export VLLM_USE_V1=0
export VLLM_USE_FLASHINFER=0
export VLLM_TORCH_COMPILE_LEVEL=0
export VLLM_BATCH_INVARIANT=0
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}

export WANDB_ENTITY=ryan-noonan-cornell-university
export WANDB_PROJECT=LMLM-Multihop

# ── Paths ─────────────────────────────────────────────────────────────────────
GPU_TYPE=""
MODEL_PATH=""
DATABASE_PATH=""
SAVE_DIR=/share/j_sun/lmlm_multihop/checkpoints/main
DATASET_NAME="hotpotqa"
NUM_GPUS=1

# ── Batch / generation dimensions ─────────────────────────────────────────────
NUM_GENERATIONS=32
NUM_DB_ROLLOUTS=4
TOTAL_BATCH_SIZE=1024

PER_DEVICE_TRAIN_BATCH_SIZE=16
PER_DEVICE_EVAL_BATCH_SIZE=32
VLLM_GPU_MEMORY_UTILIZATION=0.6
VLLM_MEM_USER_OVERRIDE=""  # set non-empty when user passes --vllm_gpu_memory_utilization

# ── Training hyperparameters ──────────────────────────────────────────────────
LOSS_TYPE="grpo"
BETA=0.0
LEARNING_RATE=1e-6
MAX_STEPS=500
NUM_TRAIN_EPOCHS=100
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
TOOLS="--tools"
TWO_PHASE=""
RETRIEVAL_THRESHOLD=0.6
REWARD_FUNC="em"
PHASE1_REWARD_TYPE="binary"
PHASE1_PROMPT_TYPE="sft"
PHASE1_DB_WEIGHT_MODE="count"
USE_CHAT_TEMPLATE=""

# ── Ablation flags ────────────────────────────────────────────────────────────
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
        --max_completion_length) MAX_COMPLETION_LENGTH="$2";  shift 2 ;;
        --vllm_max_model_length) VLLM_MAX_MODEL_LENGTH="$2";  shift 2 ;;
        --output_dir_suffix)     OUTPUT_DIR_SUFFIX="$2";      shift 2 ;;
        --train_size)            TRAIN_SIZE="$2";             shift 2 ;;
        --total_batch_size)      TOTAL_BATCH_SIZE="$2";       shift 2 ;;
        --per_device_batch_size) PER_DEVICE_TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --num_generations)       NUM_GENERATIONS="$2";        shift 2 ;;
        --num_db_rollouts)       NUM_DB_ROLLOUTS="$2";        shift 2 ;;
        --learning_rate)         LEARNING_RATE="$2";          shift 2 ;;
        --retrieval_threshold)   RETRIEVAL_THRESHOLD="$2";    shift 2 ;;
        --top_k)                 TOP_K="$2";                  shift 2 ;;
        --save_steps)            SAVE_STEPS="$2";             shift 2 ;;
        --reward_func)           REWARD_FUNC="$2";            shift 2 ;;
        --phase1_reward_type)    PHASE1_REWARD_TYPE="$2";     shift 2 ;;
        --phase1_prompt_type)    PHASE1_PROMPT_TYPE="$2";     shift 2 ;;
        --phase1_db_weight_mode) PHASE1_DB_WEIGHT_MODE="$2";  shift 2 ;;
        --vllm_gpu_memory_utilization)
            VLLM_GPU_MEMORY_UTILIZATION="$2"
            VLLM_MEM_USER_OVERRIDE="1"
            shift 2 ;;
        --two_phase)             TWO_PHASE="--two_phase";     shift 1 ;;
        --use_chat_template)     USE_CHAT_TEMPLATE="--use_chat_template"; shift 1 ;;
        --no_tools)              TOOLS="";                    shift 1 ;;
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
        --debug)                 DEBUG=1;                     shift 1 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ── GPU-type presets ──────────────────────────────────────────────────────────
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
ACCEL_CONFIG="configs/accelerate/multi_gpu_${NUM_GPUS}.yaml"

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
    # Diff vs mx253's script: bumped 0.15 -> 0.22. Observed rank-0 mem ~80% at
    # 0.15 (carrying the ref-policy copy on top of vLLM+training); we still
    # have ~14 GB headroom on rank 0. 0.22 adds ~6.5 GB per rank, taking rank
    # 0 to ~87% (safe) and giving vLLM ~21 GB KV cache instead of ~14 GB.
    PER_DEVICE_TRAIN_BATCH_SIZE=8
    VLLM_GPU_MEMORY_UTILIZATION=0.22
fi

# Re-apply user override (must come AFTER the GPU_TYPE block so the CLI value sticks)
if [ -n "${VLLM_MEM_USER_OVERRIDE}" ]; then
    : # VLLM_GPU_MEMORY_UTILIZATION already set from the arg parser
fi

if [ -n "${DEBUG}" ]; then
    echo "Debug mode enabled"
    TRAIN_SIZE=1000
    EVAL_SIZE=10
fi

CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
export CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BATCH_SIZE / (PER_DEVICE_TRAIN_BATCH_SIZE * NUM_GPUS)))
if [ "$((GRADIENT_ACCUMULATION_STEPS * PER_DEVICE_TRAIN_BATCH_SIZE * NUM_GPUS))" -ne "${TOTAL_BATCH_SIZE}" ]; then
    echo "Error: TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE} is not divisible by PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE} * NUM_GPUS=${NUM_GPUS}" >&2
    exit 1
fi
echo "  GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS} (= ${TOTAL_BATCH_SIZE} / (${PER_DEVICE_TRAIN_BATCH_SIZE} * ${NUM_GPUS}))"

B=$((TOTAL_BATCH_SIZE / NUM_GENERATIONS))
M=$(((NUM_GENERATIONS - NUM_DB_ROLLOUTS) / NUM_DB_ROLLOUTS))

OUTPUT_DIR="${SAVE_DIR}/${MODEL_PATH##*/}-${LOSS_TYPE}-tbs${TOTAL_BATCH_SIZE}-N${NUM_GENERATIONS}-K${NUM_DB_ROLLOUTS}-B${B}-M${M}-b${BETA}-lr${LEARNING_RATE}-step${MAX_STEPS}-n${TRAIN_SIZE}-${REWARD_FUNC}"
if [ -n "${TWO_PHASE}" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}-2ph"
    [ "${PHASE1_REWARD_TYPE}" != "binary" ] && OUTPUT_DIR="${OUTPUT_DIR}-rw${PHASE1_REWARD_TYPE}"
    OUTPUT_DIR="${OUTPUT_DIR}-pr${PHASE1_PROMPT_TYPE}-w${PHASE1_DB_WEIGHT_MODE}"
fi
OUTPUT_DIR="${OUTPUT_DIR}-th${RETRIEVAL_THRESHOLD}-topk${TOP_K}"

[ "${USE_ADAPTIVE_K}" != "True" ] && OUTPUT_DIR="${OUTPUT_DIR}-nak"
[ -n "${TIER_PATH}" ]             && OUTPUT_DIR="${OUTPUT_DIR}-tier${TIER_MIN_SCORE}_${TIER_MAX_SCORE}"
[ -n "${CURRICULUM}" ]            && OUTPUT_DIR="${OUTPUT_DIR}-curric"
[ -n "${USE_INVERSES}" ]          && OUTPUT_DIR="${OUTPUT_DIR}-inv"
[ -n "${VANILLA_GRPO}" ]          && OUTPUT_DIR="${OUTPUT_DIR}-vanilla"
[ -n "${DEBUG}" ]                 && OUTPUT_DIR="${OUTPUT_DIR}-debug"
[ -n "${OUTPUT_DIR_SUFFIX}" ]     && OUTPUT_DIR="${OUTPUT_DIR}${OUTPUT_DIR_SUFFIX}"

[ "${USE_ADAPTIVE_K}" = "True" ] && ADAPTIVE_K="--adaptive_k" || ADAPTIVE_K=""

LAST_CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
RESUME_FROM_CHECKPOINT=${LAST_CKPT:+"--resume_from_checkpoint=${LAST_CKPT}"}

echo "Starting GRPO training (lz586 launcher, mx253 grpo_train.py):"
echo "  Model:                       ${MODEL_PATH}"
echo "  Output:                      ${OUTPUT_DIR}"
echo "  GPUs:                        ${NUM_GPUS} (${GPU_TYPE:-default})"
echo "  Two-phase:                   ${TWO_PHASE:-off}"
echo "  per_device_train_batch_size: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "  learning_rate:               ${LEARNING_RATE}"
echo "  vllm_gpu_memory_utilization: ${VLLM_GPU_MEMORY_UTILIZATION}"
echo "  Checkpoint:                  ${RESUME_FROM_CHECKPOINT:-none}"

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
  --vllm_max_model_length=${VLLM_MAX_MODEL_LENGTH:-4096} \
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
  $([ -n "${TIER_PATH}" ] && echo "--tier_min_score=${TIER_MIN_SCORE} --tier_max_score=${TIER_MAX_SCORE}") \
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
