#!/bin/bash
#
# Eval script for first N questions from GRPO training order
# Automatically simulates GRPO dataloader to get exact training order
#
# Usage: bash ryan-scripts/eval_first_n_grpo.sh
#

# ============================================================================
# Configuration Variables
# ============================================================================

# GRPO Dataloader Configuration
NUM_QUESTIONS=100                    # How many questions to evaluate
GRPO_NUM_GENERATIONS=8               # num_generations from GRPO training
GRPO_BATCH_SIZE=1                    # batch_size for RepeatSampler
GRPO_SHUFFLE=True                    # shuffle_dataset from GRPO
GRPO_SEED=42                         # Random seed
GRPO_TRAIN_SIZE=7000                 # Total training set size

# Model Configuration
MODEL_PATH=/share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed-grpo-B1-K4-M4-bs16-s8-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak/checkpoint-3066

# Dataset Configuration
DATASET=hotpotqa
SPLIT=train
SUB_SPLIT=train
SEED=42

# Method Configuration
METHOD=lmlm
PHASE_1=""                           # Empty = build database via run_phase_1.py
DATABASE_PATH=""                     # Empty = auto-build from questions
USE_INVERSES="--use-inverses"
TOP_K=4
SIMILARITY_THRESHOLD=0.6

# Output Configuration
OUTPUT_DIR=./first-n-eval-output
SAVE_VERSION="first_${NUM_QUESTIONS}_grpo_order"
SETTING=distractor
MAX_TOKENS=1024
SAVE_EVERY=32

# ============================================================================
# Detect GPU and Set Batch Size
# ============================================================================

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

# Check for model flags
if [[ "${MODEL_PATH}" == *"-nak"* ]]; then
    ADAPTIVE_K=""
else
    ADAPTIVE_K="--adaptive-k"
fi

if [[ "${MODEL_PATH}" == *"return_triplets"* ]] || [[ "${MODEL_PATH}" == *"-th-3"* ]]; then
    RETURN_TRIPLETS="--return-triplets"
else
    RETURN_TRIPLETS=""
fi

# ============================================================================
# Generate Temporary Questions File from GRPO Dataloader Order
# ============================================================================

TEMP_QUESTIONS_FILE=$(mktemp /tmp/grpo_questions_XXXXXX.json)
echo ""
echo "=========================================="
echo "Generating first ${NUM_QUESTIONS} questions from GRPO training order"
echo "=========================================="
echo "GRPO Dataloader Config:"
echo "  Training set size: ${GRPO_TRAIN_SIZE}"
echo "  Num generations: ${GRPO_NUM_GENERATIONS}"
echo "  Batch size: ${GRPO_BATCH_SIZE}"
echo "  Shuffle: ${GRPO_SHUFFLE}"
echo "  Seed: ${GRPO_SEED}"
echo "  Questions to extract: ${NUM_QUESTIONS}"
echo ""
echo "Temp file: ${TEMP_QUESTIONS_FILE}"

python -c "
import json
import torch
from data import get_dataset

# Simulate RepeatSampler
def simulate_repeat_sampler(num_samples, mini_repeat_count, batch_size, shuffle=True, seed=42):
    if shuffle:
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)
        indexes = torch.randperm(num_samples, generator=generator).tolist()
    else:
        indexes = list(range(num_samples))

    indexes = [indexes[i : i + batch_size] for i in range(0, len(indexes), batch_size)]
    indexes = [chunk for chunk in indexes if len(chunk) == batch_size]

    result = []
    for chunk in indexes:
        for index in chunk:
            for _ in range(mini_repeat_count):
                result.append(index)
    return result

# Load dataset
print('Loading dataset...')
train_dataset = get_dataset(
    name='${DATASET}',
    setting='${SETTING}',
    split='${SPLIT}',
    sub_split='${SUB_SPLIT}',
    limit=${GRPO_TRAIN_SIZE},
    seed=${GRPO_SEED}
)
print(f'Dataset loaded: {len(train_dataset)} examples')

# Simulate sampler
print('Simulating GRPO RepeatSampler...')
sampler_indices = simulate_repeat_sampler(
    num_samples=len(train_dataset),
    mini_repeat_count=${GRPO_NUM_GENERATIONS},
    batch_size=${GRPO_BATCH_SIZE},
    shuffle=${GRPO_SHUFFLE},
    seed=${GRPO_SEED}
)

# Extract unique indices (every Nth)
unique_indices = sampler_indices[::${GRPO_NUM_GENERATIONS}]
first_n_indices = unique_indices[:${NUM_QUESTIONS}]

print(f'Extracting first ${NUM_QUESTIONS} questions in GRPO training order...')

# Build eval-compatible JSON
eval_questions = []
for dataset_idx in first_n_indices:
    ex = train_dataset[dataset_idx]
    eval_questions.append({
        'id': ex.get('_id') or ex.get('id'),
        'question': ex['question'],
        'answers': ex['answers'],
        'supporting_facts': ex.get('supporting_facts', []),
        'contexts': ex.get('contexts', []),
        'golden_contexts': ex.get('golden_contexts', []),
    })

# Save to temp file
with open('${TEMP_QUESTIONS_FILE}', 'w', encoding='utf-8') as f:
    json.dump(eval_questions, f, indent=2, ensure_ascii=False)

print(f'Saved {len(eval_questions)} questions to temp file')
print(f'First question ID: {eval_questions[0][\"id\"]}')
"

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate questions file"
    rm -f "${TEMP_QUESTIONS_FILE}"
    exit 1
fi

echo "✓ Questions generated successfully"

# ============================================================================
# Build Database (Phase 1)
# ============================================================================

mkdir -p "${OUTPUT_DIR}"

if [ -n "${PHASE_1}" ]; then
    echo ""
    echo "=========================================="
    echo "Skipping database build: Using --phase-1 mode"
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
        rm -f "${TEMP_QUESTIONS_FILE}"
        exit 1
    fi
else
    TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
    MODEL_NAME=$(basename "${MODEL_PATH}")
    BUILT_DATABASE="${OUTPUT_DIR}/phase1_database_${MODEL_NAME}_first${NUM_QUESTIONS}_${TIMESTAMP}.json"

    echo ""
    echo "=========================================="
    echo "Phase 1: Building database"
    echo "=========================================="
    echo "Model: ${MODEL_NAME}"
    echo "Questions: ${NUM_QUESTIONS} (from GRPO training order)"
    echo "Output: ${BUILT_DATABASE}"
    echo ""

    python scripts/run_phase_1.py \
        --model_path="${MODEL_PATH}" \
        --dataset="${DATASET}" \
        --split="${SPLIT}" \
        --seed=${SEED} \
        --batch_size=${BATCH_SIZE} \
        --output_file="${BUILT_DATABASE}" \
        --questions_file="${TEMP_QUESTIONS_FILE}"

    if [ $? -ne 0 ] || [ ! -f "${BUILT_DATABASE}" ]; then
        echo "Error: Database build failed"
        rm -f "${TEMP_QUESTIONS_FILE}"
        exit 1
    fi

    echo "✓ Database created successfully: ${BUILT_DATABASE}"
    DATABASE_ARG="--database-path ${BUILT_DATABASE}"
fi

# ============================================================================
# Run Evaluation (Phase 2)
# ============================================================================

echo ""
echo "=========================================="
echo "Running Evaluation"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Questions: ${NUM_QUESTIONS} (from GRPO training order)"
echo "Method: ${METHOD}"
echo "Batch size: ${BATCH_SIZE}"
echo "TOP_K: ${TOP_K}"
echo "Similarity threshold: ${SIMILARITY_THRESHOLD}"
echo "Database: ${DATABASE_ARG:-dynamic (phase-1)}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

python src/eval_multihop.py \
    --questions-file ${TEMP_QUESTIONS_FILE} \
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

EVAL_EXIT_CODE=$?

# ============================================================================
# Cleanup
# ============================================================================

echo ""
echo "Cleaning up temporary files..."
rm -f "${TEMP_QUESTIONS_FILE}"

if [ ${EVAL_EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Evaluation completed successfully!"
    echo "Results saved to: ${OUTPUT_DIR}"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Evaluation failed with exit code ${EVAL_EXIT_CODE}"
    echo "=========================================="
    exit ${EVAL_EXIT_CODE}
fi
