#!/bin/bash

# Evaluate a custom Qwen3 Search-R1 checkpoint with the Hermes/ICL3Hop
# protocol used during training. This is intentionally separate from
# eval_search_r1.slurm, which targets the released Qwen2.5 checkpoints and
# their literal <search>/<information> protocol.
#
# Required environment variables:
#   MODEL_PATH, DATASET, NUM_SAMPLES, RETRIEVAL_URL, OUTPUT_DIR
# PopQA also requires POPQA_CORPUS_PATH.
#
# Example:
#   sbatch --export=ALL,MODEL_PATH=/path/to/model,DATASET=popqa,NUM_SAMPLES=1399,\
#RETRIEVAL_URL=http://retriever:8000/retrieve,OUTPUT_DIR=/path/to/results \
#     scripts/lx_scripts/eval_search_r1_hermes.slurm

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the merged Qwen3 checkpoint}"
: "${DATASET:?Set DATASET}"
: "${NUM_SAMPLES:?Set NUM_SAMPLES}"
: "${RETRIEVAL_URL:?Set RETRIEVAL_URL to a compatible /retrieve endpoint}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

if [[ "${DATASET}" == "popqa" ]]; then
    : "${POPQA_CORPUS_PATH:?Set POPQA_CORPUS_PATH for the fixed PopQA corpus}"
    export POPQA_CORPUS_PATH
fi

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-.}}"
SPLIT="${SPLIT:-dev}"
START_INDEX="${START_INDEX:-0}"
SEED="${SEED:-42}"
RETRIEVAL_BATCH_SIZE="${RETRIEVAL_BATCH_SIZE:-512}"
MODEL_TAG="$(basename "${MODEL_PATH}")"
RUN_KEY="${MODEL_TAG}_${DATASET}_${SPLIT}_start${START_INDEX}_n${NUM_SAMPLES}_icl3hop"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/${RUN_KEY}.json}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${OUTPUT_DIR}/${RUN_KEY}.checkpoint.json}"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"

python rebuttal-search-r1/eval_search_r1_hermes.py \
    --model_path "${MODEL_PATH}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --start_index "${START_INDEX}" \
    --num_samples "${NUM_SAMPLES}" \
    --seed "${SEED}" \
    --max_turns 5 \
    --max_response_length 2048 \
    --max_model_len 8192 \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 4 \
    --topk 3 \
    --max_num_seqs 16 \
    --gpu_memory_utilization 0.45 \
    --prompt_variant icl3hop \
    --enable_thinking false \
    --max_tool_response_length 1024 \
    --tool_response_truncate_side left \
    --retrieval_batch_size "${RETRIEVAL_BATCH_SIZE}" \
    --retrieval_url "${RETRIEVAL_URL}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --resume
