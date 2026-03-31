#!/bin/bash

# GRPO Training Script for LMLM Multi-Hop QA

# ── Environment ───────────────────────────────────────────────────────────────
export VLLM_USE_V1=0               # Force V0 engine; V1 has CUDA illegal memory access bug
export VLLM_USE_FLASHINFER=0
export VLLM_TORCH_COMPILE_LEVEL=0  # Disable torch.compile to avoid inductor cache dir assertion errors
export VLLM_BATCH_INVARIANT=0
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}

# Redirect Ray temp/spill dir away from /tmp (only 15G) to home dir
export RAY_TMPDIR=/home/lz586/ray_tmp
mkdir -p /home/lz586/ray_tmp

export WANDB_ENTITY=ryan-noonan-cornell-university
export WANDB_PROJECT=LMLM-Multihop

# ── Paths ─────────────────────────────────────────────────────────────────────
GPU_TYPE=""
MODEL_PATH=/share/j_sun/rtn27/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k
DATABASE_PATH=""  # Not used in two-phase mode
SAVE_DIR=/share/j_sun/lmlm_multihop/checkpoints/debug
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
NUM_GENERATIONS=32           # N
NUM_DB_ROLLOUTS=4            # K  (N must be divisible by K)
TOTAL_BATCH_SIZE=1024

PER_DEVICE_TRAIN_BATCH_SIZE=16
PER_DEVICE_EVAL_BATCH_SIZE=32
VLLM_GPU_MEMORY_UTILIZATION=0.6

# ── Training hyperparameters ──────────────────────────────────────────────────
LOSS_TYPE="grpo"
BETA=0.0
LEARNING_RATE=1e-6
MAX_STEPS=500
NUM_TRAIN_EPOCHS=100         # High ceiling; MAX_STEPS takes effect first
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

# ── Retrieval & reward ────────────────────────────────────────────────────────
RETRIEVAL_THRESHOLD=0.6
REWARD_FUNC="em"
USE_ADAPTIVE_K=False
USE_INVERSES=False

# ── Two-phase settings ────────────────────────────────────────────────────────
PHASE1_REWARD_TYPE="binary"
PHASE1_PROMPT_TYPE="sft"
PHASE1_DB_WEIGHT_MODE="count"  # none | fixed[_<w>] | dynamic | count | count_dynamic
USE_CHAT_TEMPLATE=False
VANILLA_GRPO=""               # set to 1 via --vanilla_grpo to enable vanilla GRPO mode

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
        --learning_rate)         LEARNING_RATE="$2";          shift 2 ;;
        --reward_func)           REWARD_FUNC="$2";            shift 2 ;;
        --retrieval_threshold)   RETRIEVAL_THRESHOLD="$2";    shift 2 ;;
        --top_k)                 TOP_K="$2";                  shift 2 ;;
        --use_adaptive_k)        USE_ADAPTIVE_K="$2";         shift 2 ;;
        --num_db_rollouts)       NUM_DB_ROLLOUTS="$2";        shift 2 ;;
        --phase1_reward_type)    PHASE1_REWARD_TYPE="$2";     shift 2 ;;
        --phase1_prompt_type)    PHASE1_PROMPT_TYPE="$2";     shift 2 ;;
        --phase1_db_weight_mode) PHASE1_DB_WEIGHT_MODE="$2";  shift 2 ;;
        --use_chat_template)     USE_CHAT_TEMPLATE=True;      shift 1 ;;
        --two_phase)             TWO_PHASE=1;                 shift 1 ;;
        --vanilla_grpo)          VANILLA_GRPO=1;              shift 1 ;;
        --debug)                 DEBUG=1;                     shift 1 ;;
        --num_generations)       NUM_GENERATIONS="$2";        shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ── GPU-type presets ──────────────────────────────────────────────────────────
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
ACCEL_CONFIG="configs/accelerate/multi_gpu_${NUM_GPUS}.yaml"  # default; overridden for large models
if [ "$GPU_TYPE" == "B200" ]; then
    if [[ "${MODEL_PATH}" == *"1.7B"* ]]; then
        # NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        PER_DEVICE_TRAIN_BATCH_SIZE=16
        LEARNING_RATE=5e-6
        VLLM_GPU_MEMORY_UTILIZATION=0.4
    elif [[ "${MODEL_PATH}" == *"4B"* ]]; then
        # NUM_GPUS=2
        LEARNING_RATE=5e-6
        PER_DEVICE_TRAIN_BATCH_SIZE=8
        VLLM_GPU_MEMORY_UTILIZATION=0.15
    elif [[ "${MODEL_PATH}" == *"8B"* ]]; then
        # NUM_GPUS=2
        LEARNING_RATE=5e-6
        # NUM_GENERATIONS=16
        PER_DEVICE_TRAIN_BATCH_SIZE=4
        VLLM_GPU_MEMORY_UTILIZATION=0.2
        ACCEL_CONFIG="configs/accelerate/fsdp_${NUM_GPUS}.yaml"

    elif [[ "${MODEL_PATH}" == *"382M"* ]]; then
        # NUM_GPUS=1
        PER_DEVICE_TRAIN_BATCH_SIZE=256
        VLLM_GPU_MEMORY_UTILIZATION=0.15
    else
        echo "Error: unsupported model size for B200 preset: ${MODEL_PATH}"
        exit 1
    fi
elif [ "$GPU_TYPE" == "H100" ]; then
    # NUM_GPUS=2
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    PER_DEVICE_TRAIN_BATCH_SIZE=8
    VLLM_GPU_MEMORY_UTILIZATION=0.15
fi

if [ -n "${DEBUG}" ]; then
    echo "Debug mode enabled"
    TRAIN_SIZE=1000
    EVAL_SIZE=10
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
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

# Multi-GPU two_phase mode: gather() ordering is fixed in lmlm_basetrainer.py
# (rewards are reordered from [GPU0_p1+p2, GPU1_p1+p2, ...] to [all_p1, all_p2]).

# ── Output directory ──────────────────────────────────────────────────────────
# Format: {model}-{loss}-tbs{total_batch}-N{num_gen}-K{db_rollouts}-B{questions/batch}-M{qa_per_db}-b{beta}-step{max_steps}-n{train_size}-{reward}[-2ph[-rw{reward_type}]-pr{prompt_type}-w{weight_mode}]-th{threshold}-topk{top_k}[-nak][-debug]
B=$((TOTAL_BATCH_SIZE / NUM_GENERATIONS))                       # unique questions per global batch
M=$(((NUM_GENERATIONS - NUM_DB_ROLLOUTS) / NUM_DB_ROLLOUTS))    # phase 2 QA rollouts per (question, DB) pair
OUTPUT_DIR="${SAVE_DIR}/${MODEL_PATH##*/}-${LOSS_TYPE}-tbs${TOTAL_BATCH_SIZE}-N${NUM_GENERATIONS}-K${NUM_DB_ROLLOUTS}-B${B}-M${M}-b${BETA}-lr${LEARNING_RATE}-step${MAX_STEPS}-n${TRAIN_SIZE}-${REWARD_FUNC}"
if [ -n "${TWO_PHASE}" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}-2ph"
    [ "${PHASE1_REWARD_TYPE}" != "binary" ] && OUTPUT_DIR="${OUTPUT_DIR}-rw${PHASE1_REWARD_TYPE}"
    OUTPUT_DIR="${OUTPUT_DIR}-pr${PHASE1_PROMPT_TYPE}-w${PHASE1_DB_WEIGHT_MODE}"
    [ -n "${VANILLA_GRPO}" ] && OUTPUT_DIR="${OUTPUT_DIR}-vanilla"
    TWO_PHASE="--two_phase"
else
    TWO_PHASE=""
fi
OUTPUT_DIR="${OUTPUT_DIR}-th${RETRIEVAL_THRESHOLD}-topk${TOP_K}"
[ -n "${DEBUG}" ] && OUTPUT_DIR="${OUTPUT_DIR}-debug"

# ── Flag resolution ───────────────────────────────────────────────────────────
# Return triples: enabled when model path encodes threshold -3 (suffix _th-3)
THRESHOLD="${MODEL_PATH##*_th}"
[ "${THRESHOLD}" = "-3" ] && RETURN_TRIPLES="--return_triples" || RETURN_TRIPLES=""

if [ "${USE_ADAPTIVE_K}" = "True" ]; then
    ADAPTIVE_K="--adaptive_k"
else
    ADAPTIVE_K=""
    OUTPUT_DIR="${OUTPUT_DIR}-nak"
fi

[ "${USE_INVERSES}" = "True" ] && INVERSES_FLAG="--use-inverses" || INVERSES_FLAG=""

# ── Resume from checkpoint ────────────────────────────────────────────────────
LAST_CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
RESUME_FROM_CHECKPOINT=${LAST_CKPT:+"--resume_from_checkpoint=${LAST_CKPT}"}

# ── Summary ───────────────────────────────────────────────────────────────────
echo "Starting GRPO training:"
echo "  Model:      ${MODEL_PATH}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  GPUs:       ${NUM_GPUS} (${GPU_TYPE:-default})"
echo "  Two-phase:  ${TWO_PHASE:-off}"
echo "  Checkpoint: ${RESUME_FROM_CHECKPOINT:-none}"

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
  --max_steps=${MAX_STEPS} \
  --save_strategy=steps \
  --save_total_limit=5 \
  --save_steps=${SAVE_STEPS} \
  --eval_strategy=steps \
  --eval_steps=${EVAL_STEPS} \
  ${RESUME_FROM_CHECKPOINT} \
  --retrieval-threshold ${RETRIEVAL_THRESHOLD} \
  ${TWO_PHASE} \
  ${RETURN_TRIPLES} \
  ${INVERSES_FLAG} \
  ${ADAPTIVE_K} \
  --reward_func=${REWARD_FUNC} \
  --phase1_reward_type=${PHASE1_REWARD_TYPE} \
  --phase1_prompt_type=${PHASE1_PROMPT_TYPE} \
  --num_db_rollouts=${NUM_DB_ROLLOUTS} \
  --phase1_db_weight_mode=${PHASE1_DB_WEIGHT_MODE} \
  $([ "${USE_CHAT_TEMPLATE}" = "True" ] && echo "--use_chat_template") \
  $([ -n "${VANILLA_GRPO}" ] && echo "--vanilla_grpo")

echo "Training completed!"

#   --eval_on_start \
