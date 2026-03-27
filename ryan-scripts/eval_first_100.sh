#!/bin/bash
#
# Eval script for first 100 training questions
# Usage: bash ryan-scripts/eval_first_100.sh
#

# Model configuration
MODEL_PATH=/share/j_sun/lmlm_multihop/checkpoints/ryan-march-sft-exp/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_2k_db_2k_qa_classic_retrieval
MODEL_PATH=/share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed-grpo-B1-K4-M4-bs16-s8-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak/checkpoint-3066

# Dataset configuration
DATASET=hotpotqa
SPLIT=train
QUESTIONS_FILE="first_100_grpo_training_order.json"  # Pre-generated file with first 100 GRPO training questions
SEED=42

# Method configuration
METHOD=lmlm
PHASE_1=""  # Set to "--phase-1" to build database dynamically from golden contexts
DATABASE_PATH=""  # Set to path of pre-built database, or leave empty to require --phase-1
USE_INVERSES="--use-inverses"
TOP_K=4
SIMILARITY_THRESHOLD=0.6

# Output configuration
OUTPUT_DIR=./first-100-eval-output
SAVE_VERSION="first_100_train"
SETTING=distractor
MAX_TOKENS=1024
SAVE_EVERY=32

# Detect GPU and set batch size
echo "Detecting GPU type..."
GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "GPU detected: ${GPU_INFO}"

if [[ "${GPU_INFO}" == *"H100"* ]]; then
    BATCH_SIZE=64
    echo "Setting batch size to 64 for H100"
elif [[ "${GPU_INFO}" == *"6000 Ada"* ]] || [[ "${GPU_INFO}" == *"RTX 6000 Ada"* ]]; then
    BATCH_SIZE=24
    echo "Setting batch size to 24 for 6000 Ada"
else
    BATCH_SIZE=16
    echo "Setting batch size to 16 (default)"
fi

# Check for adaptive-k flag in model path
if [[ "${MODEL_PATH}" == *"-nak"* ]]; then
    ADAPTIVE_K=""
else
    ADAPTIVE_K="--adaptive-k"
fi

# Check for return-triplets flag in model path
if [[ "${MODEL_PATH}" == *"return_triplets"* ]] || [[ "${MODEL_PATH}" == *"-th-3"* ]]; then
    RETURN_TRIPLETS="--return-triplets"
else
    RETURN_TRIPLETS=""
fi

echo ""
echo "=========================================="
echo "Evaluation Configuration"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Questions file: ${QUESTIONS_FILE}"
echo "Dataset: ${DATASET} (for agent initialization)"
echo "Split: ${SPLIT} (for agent initialization)"
echo "Method: ${METHOD}"
echo "Batch size: ${BATCH_SIZE}"
echo "TOP_K: ${TOP_K}"
echo "Similarity threshold: ${SIMILARITY_THRESHOLD}"
echo "Phase-1 mode: ${PHASE_1}"
echo "Use inverses: ${USE_INVERSES}"
echo "Adaptive K: ${ADAPTIVE_K}"
echo "Return triplets: ${RETURN_TRIPLETS}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# Check if questions file exists
if [ ! -f "${QUESTIONS_FILE}" ]; then
    echo "ERROR: Questions file not found: ${QUESTIONS_FILE}"
    echo "Please run: python create_eval_questions_json.py"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Build database using run_phase_1.py (same logic as grid eval script)
if [ -n "${PHASE_1}" ]; then
    echo ""
    echo "=========================================="
    echo "Skipping Phase 1 database build: Using --phase-1 mode"
    echo "Database will be built dynamically from golden contexts"
    echo "=========================================="
    DATABASE_ARG=""
elif [ -n "${DATABASE_PATH}" ]; then
    echo ""
    echo "=========================================="
    echo "Using pre-built database: ${DATABASE_PATH}"
    echo "=========================================="
    DATABASE_ARG="--database-path ${DATABASE_PATH}"

    if [ ! -f "${DATABASE_PATH}" ]; then
        echo "Error: Database file ${DATABASE_PATH} does not exist"
        exit 1
    fi
else
    # Build database from the questions file
    TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
    MODEL_NAME=$(basename "${MODEL_PATH}")
    BUILT_DATABASE="${OUTPUT_DIR}/phase1_database_${MODEL_NAME}_first100_${TIMESTAMP}.json"

    echo ""
    echo "=========================================="
    echo "Phase 1: Building database for first 100 questions"
    echo "Model: ${MODEL_NAME}"
    echo "Output: ${BUILT_DATABASE}"
    echo "=========================================="

    python scripts/run_phase_1.py \
        --model_path="${MODEL_PATH}" \
        --dataset="${DATASET}" \
        --split="${SPLIT}" \
        --seed=${SEED} \
        --batch_size=${BATCH_SIZE} \
        --output_file="${BUILT_DATABASE}" \
        --questions_file="${QUESTIONS_FILE}"

    if [ ! -f "${BUILT_DATABASE}" ]; then
        echo "Error: Database file ${BUILT_DATABASE} was not created"
        exit 1
    fi

    echo "Database created successfully: ${BUILT_DATABASE}"
    DATABASE_ARG="--database-path ${BUILT_DATABASE}"
fi
echo ""

# Run evaluation
python src/eval_multihop.py \
    --questions-file ${QUESTIONS_FILE} \
    --model-path ${MODEL_PATH} \
    ${DATABASE_ARG} \
    --method ${METHOD} \
    --max-tokens ${MAX_TOKENS} \
    --batch-size ${BATCH_SIZE} \
    --output-dir ${OUTPUT_DIR}/ \
    --save-version ${SAVE_VERSION} \
    --dataset ${DATASET} \
    --split ${SPLIT} \
    --setting ${SETTING} \
    --seed ${SEED} \
    --save-every ${SAVE_EVERY} \
    --top-k ${TOP_K} \
    --similarity-threshold ${SIMILARITY_THRESHOLD} \
    ${ADAPTIVE_K} \
    ${RETURN_TRIPLETS} \
    ${USE_INVERSES} \
    ${PHASE_1} \
    --eval

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="
