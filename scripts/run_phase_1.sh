#!/bin/bash

# Phase 1 Database Generation Script
# Generates LMLM database triplets from model completions

# Default values
MODEL_PATH="/share/j_sun/rtn27/checkpoints/lmlm_multi_hop//Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed"
DATASET="hotpotqa"
SPLIT="dev"
SEED=42
NB_EXAMPLES=1000
BATCH_SIZE=32
OUTPUT_FILE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --nb_examples)
            NB_EXAMPLES="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--model_path PATH] [--dataset NAME] [--split SPLIT] [--seed SEED] [--nb_examples N] [--batch_size N] [--output_file PATH]"
            exit 1
            ;;
    esac
done

echo "Starting Phase 1 database generation with:"
echo "  Model: ${MODEL_PATH}"
echo "  Dataset: ${DATASET}"
echo "  Split: ${SPLIT}"
echo "  Seed: ${SEED}"
echo "  Number of examples: ${NB_EXAMPLES}"
echo "  Batch size: ${BATCH_SIZE}"
if [ -n "${OUTPUT_FILE}" ]; then
    echo "  Output file: ${OUTPUT_FILE}"
    OUTPUT_FILE_ARG="--output_file=${OUTPUT_FILE}"
else
    echo "  Output file: (auto-generated)"
    OUTPUT_FILE_ARG=""
fi

python scripts/run_phase_1.py \
    --model_path="${MODEL_PATH}" \
    --dataset="${DATASET}" \
    --split="${SPLIT}" \
    --seed=${SEED} \
    --NB_EXAMPLES=${NB_EXAMPLES} \
    --batch_size=${BATCH_SIZE} \
    ${OUTPUT_FILE_ARG}

echo "Phase 1 generation completed!"
