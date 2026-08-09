#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=jjs533
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/rtn27/qwen25_%x_%j.out
#SBATCH --error=/home/rtn27/qwen25_%x_%j.err

set -euo pipefail

: "${METHOD:?Set METHOD}"
: "${DATASET:?Set DATASET}"
: "${RUN_TAG:?Set RUN_TAG}"

REPO_DIR="${REPO_DIR:-/home/rtn27/Multi-Hop-Reasoning-qwen25-table3}"
OUTPUT_DIR="${OUTPUT_DIR:-/share/j_sun/rtn27/qwen25_table3_eval/${RUN_TAG}}"
POPQA_ARTIFACT_DIR="${POPQA_ARTIFACT_DIR:-/share/j_sun/rtn27/popqa_longtail_1399}"
POPQA_CORPUS_PATH="${POPQA_ARTIFACT_DIR}/popqa_longtail_contexts.jsonl"

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/rtn27/multi-hop-reasoning
cd "${REPO_DIR}"

SETTING=distractor
SPLIT=dev
NUM_SAMPLES=1000
EXTRA_ARGS=()
if [[ "${DATASET}" == "popqa" ]]; then
    SETTING=long_tail
    SPLIT=test
    NUM_SAMPLES=1399
    export POPQA_CORPUS_PATH
    if [[ "${METHOD}" == "rag" ]]; then
        EXTRA_ARGS+=(--rag-corpus-path "${POPQA_CORPUS_PATH}")
    fi
fi

if [[ "${METHOD}" == "rag" ]]; then
    BATCH_SIZE=1
    EXTRA_ARGS+=(--rag-k 4)
elif [[ "${METHOD}" == "direct" ]]; then
    BATCH_SIZE=32
else
    echo "Unsupported METHOD=${METHOD}" >&2
    exit 2
fi

echo "code_commit=$(git rev-parse HEAD)"
echo "method=${METHOD} model=qwen2.5-3b dataset=${DATASET} setting=${SETTING} n=${NUM_SAMPLES}"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader

python src/eval_multihop.py \
    --method "${METHOD}" \
    --model qwen2.5-3b \
    --dataset "${DATASET}" \
    --dataset-source hf \
    --setting "${SETTING}" \
    --split "${SPLIT}" \
    --seed 42 \
    --start-index 0 \
    --total-count "${NUM_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --max-tokens 1024 \
    --save-every 64 \
    --output-dir "${OUTPUT_DIR}" \
    --save-version "_${RUN_TAG}" \
    --eval \
    "${EXTRA_ARGS[@]}"
