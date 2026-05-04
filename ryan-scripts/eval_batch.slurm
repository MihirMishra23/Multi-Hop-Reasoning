#!/bin/bash
#SBATCH --mail-user=ryan.noonan@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=eval-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
##SBATCH --partition=jjs533
#SBATCH --constraint='6000ada'
#SBATCH --time=8:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# Activate conda env from /scratch (faster than shared conda)
eval "$(conda shell.bash hook)"
if [[ "${SLURM_JOB_PARTITION}" == "jjs533" ]]; then
    conda activate /scratch/rtn27/multi-hop-reasoning
else
    conda activate multi-hop-reasoning
fi

# Required parameters (passed via --export)
: "${DATASET:?DATASET environment variable must be set}"
: "${NUM_SAMPLES:?NUM_SAMPLES environment variable must be set}"
: "${SEED:?SEED environment variable must be set}"
: "${MODEL_PATH:?MODEL_PATH environment variable must be set}"

# Default parameters (can be overridden via --export)
TWO_PHASE_METHOD="${TWO_PHASE_METHOD:-two_phase}"
SPLIT="${SPLIT:-dev}"
SAVE_VERSION="${SAVE_VERSION:-concat-all}"
USE_TRAIN_PARAMS="${USE_TRAIN_PARAMS:---use-train-params}"
CONCAT_ALL_DB="${CONCAT_ALL_DB:---concat-all-db}"
USE_CONTEXTS="${USE_CONTEXTS:-all}"
OUTPUT_DIR="${OUTPUT_DIR:-./march-31-trivia-qa-pop-qa-final}"
SETTING="${SETTING:-distractor}"
CONFIQA_SETTING="${CONFIQA_SETTING:-}"

echo "=========================================="
echo "Starting Eval..."
echo "Dataset: ${DATASET}"
echo "Num Samples: ${NUM_SAMPLES}"
echo "Seed: ${SEED}"
echo "Model: ${MODEL_PATH}"
echo "=========================================="

# Automatically set setting to 'qa-sm' for synthworlds dataset
if [[ "${DATASET}" == "synthworlds" ]]; then
    DATASET_SETTING="qa-sm"
    DATASET_USE_CONTEXTS="golden"
    echo "NOTE: Automatically setting --use-contexts=golden for synthworlds dataset (only golden contexts are available)"
else
    DATASET_SETTING="${SETTING}"
    DATASET_USE_CONTEXTS="${USE_CONTEXTS}"
fi

echo "========== Evaluating ${MODEL_PATH} on ${DATASET} (setting=${DATASET_SETTING}, use-contexts=${DATASET_USE_CONTEXTS}) =========="

# Build confiqa-setting argument if provided
CONFIQA_SETTING_FLAG=""
if [ -n "${CONFIQA_SETTING:-}" ]; then
    CONFIQA_SETTING_FLAG="--confiqa-setting ${CONFIQA_SETTING}"
fi

bash /home/rtn27/Multi-Hop-Reasoning/ryan-scripts/eval_multihop.sh \
    --model_path "${MODEL_PATH}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --num_samples ${NUM_SAMPLES} \
    --save_version "${SAVE_VERSION}" \
    --method "${TWO_PHASE_METHOD}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed ${SEED} \
    ${USE_TRAIN_PARAMS} \
    ${CONCAT_ALL_DB} \
    --use-contexts "${DATASET_USE_CONTEXTS}" \
    --setting "${DATASET_SETTING}" \
    ${CONFIQA_SETTING_FLAG}

echo "Evaluation complete!"
