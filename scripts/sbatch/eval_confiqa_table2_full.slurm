#!/bin/bash
# Submit one independent job per model/setting by setting MODE, MODEL_PATH,
# and CONFIQA_SETTING. MODE is one of: kbevo, searchr1.

#SBATCH --job-name=confiqa_t2_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=default_partition
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --requeue
#SBATCH --output=/home/rtn27/slurm_output/%x_%j.out
#SBATCH --error=/home/rtn27/slurm_output/%x_%j.err

set -euo pipefail

MODE="${MODE:?Set MODE to kbevo or searchr1}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH}"
CONFIQA_SETTING="${CONFIQA_SETTING:?Set CONFIQA_SETTING}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
SEED="${SEED:-42}"
RUN_ROOT="${RUN_ROOT:-/share/j_sun/lmlm_multihop/results/confiqa_table2_full}"
PYTHON="${PYTHON:-/home/rtn27/.conda/envs/multi-hop-reasoning/bin/python}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR}}"

cd "${REPO_DIR}"
mkdir -p "${RUN_ROOT}" /home/rtn27/slurm_output
export VLLM_CACHE_ROOT="/tmp/vllm_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_${SLURM_JOB_ID}"
mkdir -p "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}"

if [[ "${MODE}" == "kbevo" ]]; then
    "${PYTHON}" src/eval_multihop.py \
        --model-path "${MODEL_PATH}" \
        --method two_phase \
        --dataset confiqa \
        --split dev \
        --confiqa-setting "${CONFIQA_SETTING}" \
        --seed "${SEED}" \
        --start-index 0 \
        --total-count "${NUM_SAMPLES}" \
        --batch-size "${EVAL_BATCH_SIZE}" \
        --save-every 1 \
        --output-dir "${RUN_ROOT}/kbevo" \
        --save-version full \
        --concat-all-db \
        --use-contexts all \
        --use-inverses \
        --top-k 4 \
        --similarity-threshold 0.6 \
        --use-train-params \
        --resume \
        --eval
elif [[ "${MODE}" == "searchr1" ]]; then
    MODEL_TAG="$(basename "${MODEL_PATH}")"
    WORK_DIR="${RUN_ROOT}/searchr1/${MODEL_TAG}/${CONFIQA_SETTING}_n${NUM_SAMPLES}"
    mkdir -p "${WORK_DIR}"
    "${PYTHON}" rebuttal-search-r1/prepare_confiqa_corpus.py \
        --setting "${CONFIQA_SETTING}" \
        --num-samples "${NUM_SAMPLES}" \
        --seed "${SEED}" \
        --output-dir "${WORK_DIR}"
    "${PYTHON}" rebuttal-search-r1/build_retrieval_index.py \
        --corpus "${WORK_DIR}/corpus.jsonl" \
        --output "${WORK_DIR}/e5_Flat.index"
    "${PYTHON}" rebuttal-search-r1/eval_search_r1_qwen25.py \
        --model-path "${MODEL_PATH}" \
        --confiqa-setting "${CONFIQA_SETTING}" \
        --num-samples "${NUM_SAMPLES}" \
        --seed "${SEED}" \
        --retrieval-corpus "${WORK_DIR}/corpus.jsonl" \
        --retrieval-index "${WORK_DIR}/e5_Flat.index" \
        --output "${WORK_DIR}/results.json"
else
    echo "Unsupported MODE=${MODE}" >&2
    exit 2
fi
