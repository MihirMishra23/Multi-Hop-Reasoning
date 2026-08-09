#!/bin/bash
# Submit one job per model/condition by setting MODE, MODEL_PATH and
# CONFIQA_CONDITION.  Slurm partition/GPU constraints are supplied by the
# launcher after checking the account's current associations.
# MODE is one of: kbevo, searchr1

#SBATCH --job-name=confiqa_t2_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/rtn27/slurm_output/%x_%j.out
#SBATCH --error=/home/rtn27/slurm_output/%x_%j.err

set -euo pipefail

MODE="${MODE:?Set MODE to kbevo or searchr1}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH}"
CONFIQA_CONDITION="${CONFIQA_CONDITION:?Set CONFIQA_CONDITION}"
QUERY_SAMPLES="${QUERY_SAMPLES:-50}"
STORE_SAMPLES="${STORE_SAMPLES:-1000}"
SEED="${SEED:-42}"
RUN_ROOT="${RUN_ROOT:-/share/j_sun/lmlm_multihop/results/confiqa_conflict_free_smoke}"
PYTHON="${PYTHON:-/home/rtn27/.conda/envs/multi-hop-reasoning/bin/python}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR}}"
export PATH="$(dirname "${PYTHON}"):${PATH}"

cd "${REPO_DIR}"
mkdir -p "${RUN_ROOT}" "${LOG_DIR:-${RUN_ROOT}/logs}"
CACHE_BASE="${CACHE_BASE:-/scratch/${USER}/confiqa_smoke_${SLURM_JOB_ID}}"
export VLLM_CACHE_ROOT="${CACHE_BASE}/vllm"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_BASE}/inductor"
export TMPDIR="${CACHE_BASE}/tmp"
mkdir -p "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" "${TMPDIR}"
set -x

if [[ "${MODE}" == "kbevo" ]]; then
    "${PYTHON}" src/eval_multihop.py \
        --model-path "${MODEL_PATH}" \
        --method two_phase \
        --dataset confiqa \
        --split dev \
        --confiqa-setting "${CONFIQA_CONDITION}" \
        --seed "${SEED}" \
        --start-index 0 \
        --total-count "${QUERY_SAMPLES}" \
        --knowledge-store-count "${STORE_SAMPLES}" \
        --batch-size "${QUERY_SAMPLES}" \
        --save-every 1 \
        --output-dir "${RUN_ROOT}/kbevo" \
        --save-version smoke50_store1000 \
        --concat-all-db \
        --use-contexts all \
        --use-inverses \
        --top-k 4 \
        --similarity-threshold 0.6 \
        --use-train-params \
        --eval
elif [[ "${MODE}" == "searchr1" ]]; then
    MODEL_TAG="$(basename "${MODEL_PATH}")"
    WORK_DIR="${RUN_ROOT}/searchr1/${MODEL_TAG}/${CONFIQA_CONDITION}_q${QUERY_SAMPLES}_store${STORE_SAMPLES}"
    mkdir -p "${WORK_DIR}"
    "${PYTHON}" rebuttal-search-r1/prepare_confiqa_corpus.py \
        --setting "${CONFIQA_CONDITION}" \
        --num-samples "${STORE_SAMPLES}" \
        --seed "${SEED}" \
        --output-dir "${WORK_DIR}"
    "${PYTHON}" rebuttal-search-r1/build_retrieval_index.py \
        --corpus "${WORK_DIR}/corpus.jsonl" \
        --output "${WORK_DIR}/e5_Flat.index" \
        --batch-size 128 \
        --device cuda
    "${PYTHON}" rebuttal-search-r1/eval_search_r1_qwen25.py \
        --model-path "${MODEL_PATH}" \
        --confiqa-setting "${CONFIQA_CONDITION}" \
        --num-samples "${QUERY_SAMPLES}" \
        --expected-store-samples "${STORE_SAMPLES}" \
        --seed "${SEED}" \
        --retrieval-corpus "${WORK_DIR}/corpus.jsonl" \
        --retrieval-index "${WORK_DIR}/e5_Flat.index" \
        --retriever-device cuda \
        --output "${WORK_DIR}/results.json"
else
    echo "Unsupported MODE=${MODE}" >&2
    exit 2
fi
