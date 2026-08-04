#!/bin/bash
#
# sbatch wrapper to run eval_search_r1.py against a merged HF checkpoint.
#
# Submit with:
#   MERGED_CKPT=/home/lz586/icl/Multi-Hop-Reasoning/rebuttal-search-r1/merged_ckpts/qwen3-1.7b-searchR1-a2a-lr1e6-step500 \
#   SPLIT=train_val1k NUM_SAMPLES=1000 \
#   sbatch --nodelist=aimi-compute-02 sbatch_eval_search_r1.slurm
#
# Required env: MERGED_CKPT
# Optional env: SPLIT (default train_val1k), NUM_SAMPLES (1000), MAX_TURNS (4),
#               TEMPERATURE (0.0), TAG (suffix for output filename).

#SBATCH -J eval_search_r1
#SBATCH -o /home/lz586/icl/slurm_output/searchr1_a2a/%x_%j.out
#SBATCH -e /home/lz586/icl/slurm_output/searchr1_a2a/%x_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 4:00:00
#SBATCH --partition=aimi
#SBATCH --gres=gpu:nvidia_b200:1

set -euo pipefail

MERGED_CKPT="${MERGED_CKPT:?Set MERGED_CKPT to a merged HF checkpoint dir}"
DATASET="${DATASET:-hotpotqa}"
SPLIT="${SPLIT:-dev}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
# Defaults below mirror the Search-R1 a2a training recipe
# (run_qwen3_1.7b_apples_to_apples.sh): T=1.0, top_p=0.95, top_k=4,
# max_response_length=2048, max_assistant_turns=5. Same distribution as
# LMLM eval's --use-train-params, so the two pipelines sample under
# identical conditions for apples-to-apples comparison.
MAX_TURNS="${MAX_TURNS:-5}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K_SAMPLE="${TOP_K_SAMPLE:-4}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TOPK_RETRIEVAL="${TOPK_RETRIEVAL:-3}"
SEED="${SEED:-42}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.45}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
TAG="${TAG:-}"

ENV_DIR="/home/lz586/.conda/envs/mem"
PROJECT_DIR="/home/lz586/icl/Multi-Hop-Reasoning/rebuttal-search-r1"
SOURCE_NODE="${SOURCE_NODE:-aimi-compute-02}"

CKPT_TAG="$(basename "${MERGED_CKPT}")"
OUTPUT_DIR="${PROJECT_DIR}/eval_results"
# Encode sampling/turns into the filename so a2a results don't collide with
# the prior T=0.0 / max_turns=4 runs in eval_results/.
SAMPLING_TAG="t${TEMPERATURE}_tp${TOP_P}_tk${TOP_K_SAMPLE}_mt${MAX_TURNS}_resp${MAX_RESPONSE_LENGTH}"
OUTPUT_PATH="${OUTPUT_DIR}/${CKPT_TAG}_${DATASET}_${SPLIT}_n${NUM_SAMPLES}_${SAMPLING_TAG}${TAG:+_${TAG}}_${SLURM_JOB_ID}.json"

echo "[$(date)] === eval sbatch start ==="
echo "Node:                 $(hostname)"
echo "JobID:                ${SLURM_JOB_ID}"
echo "MERGED_CKPT:          ${MERGED_CKPT}"
echo "DATASET:              ${DATASET}"
echo "SPLIT:                ${SPLIT}"
echo "NUM_SAMPLES:          ${NUM_SAMPLES}"
echo "MAX_TURNS:            ${MAX_TURNS}"
echo "TEMPERATURE:          ${TEMPERATURE}"
echo "TOP_P:                ${TOP_P}"
echo "TOP_K_SAMPLE:         ${TOP_K_SAMPLE}"
echo "MAX_RESPONSE_LENGTH:  ${MAX_RESPONSE_LENGTH}"
echo "MAX_MODEL_LEN:        ${MAX_MODEL_LEN}"
echo "TOPK_RETRIEVAL:       ${TOPK_RETRIEVAL}"
echo "SEED:                 ${SEED}"
echo "OUTPUT:               ${OUTPUT_PATH}"
nvidia-smi -L

# ----- 1. Env (NFS-shared mem env: has vllm 0.10.2, transformers, datasets) -----
[ -x "${ENV_DIR}/bin/python" ] || { echo "mem env missing at ${ENV_DIR}"; exit 1; }
export PATH="${ENV_DIR}/bin:${PATH}"
python3 --version
python3 -c "import vllm; print('vllm', vllm.__version__)"

# Per-job vLLM/inductor cache to avoid a race when multiple eval jobs land
# on the same node and hash the same model config to a shared cache dir
# under ~/.cache/vllm/torch_compile_cache/. Symptom without this:
# FileNotFoundError on the atomic rename, bare AssertionError in
# _initialize_kv_caches, or "Stale file handle" on NFS-backed homes.
export VLLM_CACHE_ROOT="/tmp/vllm_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_${SLURM_JOB_ID}"
mkdir -p "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}"
echo "[$(date)] VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"

# ----- 2. Retrieval server (reuse if up, else launch) -----
cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"
RETRIEVAL_LOG="${PROJECT_DIR}/logs/retrieval_eval_${SLURM_JOB_ID}.log"
LOCK_FILE="/tmp/retrieval_${USER}.lock"
RETRIEVAL_LAUNCHED_BY_US=0

# Pick a python that has the retrieval-server deps (faiss, fastapi, sentence-
# transformers, e5). Prefer the node-local scratch env on aimi nodes, else fall
# back to the NFS-mounted mem env (which also has all deps). This lets the
# eval run on any GPU partition, not just where the scratch env exists.
if [ -x "/scratch/${USER}/envs/searchr1-sgl/bin/python" ]; then
    export ENV_PYTHON="/scratch/${USER}/envs/searchr1-sgl/bin/python"
else
    export ENV_PYTHON="${ENV_DIR}/bin/python"
fi
echo "[$(date)] retrieval server will use: ${ENV_PYTHON}"

# Fast path: if retrieval is already up, skip the lock entirely. This is the
# common case (training jobs leave their retrieval running). Avoids the
# fd-inheritance bug where launch_retrieval_verl.sh inherited the lock fd
# and held it after our subshell exited, blocking later eval jobs.
if curl -sf -m 2 http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "[$(date)] retrieval already up, reusing"
else
    (
        exec 200>"${LOCK_FILE}"
        flock 200
        # Double-check after acquiring the lock (someone else may have started it).
        if curl -sf -m 2 http://127.0.0.1:8000/health > /dev/null 2>&1; then
            echo "RETRIEVAL_LAUNCHED_BY_US=0"
        else
            echo "RETRIEVAL_LAUNCHED_BY_US=1"
            # exec 200>&- inside the forked child closes the lock fd so the
            # retrieval server doesn't inherit it. setsid puts it in its own
            # session so SIGHUP doesn't reach it.
            (
                exec 200>&-
                CUDA_VISIBLE_DEVICES=0 setsid bash launch_retrieval_verl.sh \
                    > "${RETRIEVAL_LOG}" 2>&1 < /dev/null &
                disown
            )
            for i in $(seq 1 60); do
                curl -sf -m 2 http://127.0.0.1:8000/health > /dev/null 2>&1 && break
                sleep 1
            done
        fi
    ) > /tmp/retrieval_status_${SLURM_JOB_ID}.txt
    eval "$(cat /tmp/retrieval_status_${SLURM_JOB_ID}.txt)" || true
fi
echo "[$(date)] retrieval state: launched_by_us=${RETRIEVAL_LAUNCHED_BY_US} pid=${RETRIEVAL_PID:-none}"
curl -sf -m 5 http://127.0.0.1:8000/health > /dev/null \
    || { echo "retrieval never became healthy — abort"; tail -50 "${RETRIEVAL_LOG}" 2>/dev/null; exit 1; }

cleanup() {
    rm -f /tmp/retrieval_status_${SLURM_JOB_ID}.txt 2>/dev/null || true
    rm -rf "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

# ----- 3. Eval -----
# gpu_memory_utilization leaves room for retrieval encoder on the same GPU.
# Sampling/length flags below mirror the Search-R1 a2a training recipe.
echo "[$(date)] Launching eval"
python3 eval_search_r1.py \
    --model_path "${MERGED_CKPT}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --num_samples "${NUM_SAMPLES}" \
    --seed "${SEED}" \
    --max_turns "${MAX_TURNS}" \
    --max_response_length "${MAX_RESPONSE_LENGTH}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K_SAMPLE}" \
    --topk "${TOPK_RETRIEVAL}" \
    --max_num_seqs "${MAX_NUM_SEQS}" \
    --gpu_memory_utilization "${GPU_MEM_UTIL}" \
    --output_path "${OUTPUT_PATH}"

echo "[$(date)] === eval done — output: ${OUTPUT_PATH} ==="
# Quick summary print
python3 -c "
import json
with open('${OUTPUT_PATH}') as f:
    d = json.load(f)
print('SUMMARY:', json.dumps(d['summary'], indent=2))
"
