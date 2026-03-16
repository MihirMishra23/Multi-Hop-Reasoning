#!/bin/bash

#SBATCH -J grpo_train
#SBATCH -o /home/as2637/lmlm_12/Multi-Hop-Reasoning/run_phase_1_%j.out
#SBATCH -e /home/as2637/lmlm_12/Multi-Hop-Reasoning/run_phase_1_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=jjs533,gpu-interactive
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|a6000
#SBATCH --requeue

# Load the required environment (conda often not in PATH under sbatch)
CONDA_BASE="${CONDA_PREFIX%/envs/*}"
[[ -z "$CONDA_BASE" ]] && CONDA_BASE="${HOME}/miniconda"
[[ ! -d "$CONDA_BASE" ]] && CONDA_BASE="${HOME}/anaconda3"
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate lmlm_multihop
fi
if ! command -v python &>/dev/null; then
    export PATH="${HOME}/miniconda/envs/lmlm_multihop/bin:${PATH}"
fi

# Phase 1 Database Generation Script
# Generates LMLM database triplets from model completions

# Default values
MODEL_PATH="/share/j_sun/lmlm_multihop/checkpoints/debug/full-overfit-Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed-grpo-g8-bs16-s8-b0.0-ep5-n1000-em_size-v2-nak/checkpoint-240"
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

# Run from repo root so imports resolve: data -> src/data
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH}"

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

python "${REPO_ROOT}/scripts/run_phase_1.py" \
    --model_path="${MODEL_PATH}" \
    --dataset="${DATASET}" \
    --split="${SPLIT}" \
    --seed=${SEED} \
    --nb_examples=${NB_EXAMPLES} \
    --batch_size=${BATCH_SIZE} \
    ${OUTPUT_FILE_ARG}

echo "Phase 1 generation completed!"
