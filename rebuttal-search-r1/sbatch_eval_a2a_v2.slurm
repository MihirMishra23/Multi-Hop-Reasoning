#!/bin/bash
#
# Eval one merged checkpoint on one (benchmark, split) with a per-benchmark
# eval-only retrieval datastore. Each job launches its OWN retrieval server
# on a unique port (no sharing) so we can run many configs in parallel
# without index conflicts.
#
# Submit with (example):
#   MERGED_CKPT=/.../merged_ckpts/c_icl_step200 \
#   DATASET=hotpotqa SPLIT=train_val1k \
#   PROMPT_VARIANT=icl3hop ENABLE_THINKING=false \
#   sbatch sbatch_eval_a2a_v2.slurm
#
# Required env: MERGED_CKPT, DATASET, SPLIT, PROMPT_VARIANT
# Optional env: ENABLE_THINKING (default false), NUM_SAMPLES (1000), TAG.

#SBATCH -J eval_a2a
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
DATASET="${DATASET:?Set DATASET}"
SPLIT="${SPLIT:?Set SPLIT}"
PROMPT_VARIANT="${PROMPT_VARIANT:?Set PROMPT_VARIANT (default|thinkingtag|icl3hop)}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"

# Sampling: training-matching (T=1.0 top_p=0.95 top_k=4), n=1 sample per question
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
EVAL_CORPORA_DIR="${PROJECT_DIR}/data/eval_corpora"

# Per-benchmark corpus + index paths. Built by prepare_eval_corpus.py and
# build_index_hotpotqa.py (with --corpus_path / --output_dir overrides).
# Override CORPUS_PATH / INDEX_PATH in env to use a different datastore
# (e.g. full wiki18: /share/nikola/wiki18/{wiki18_100w.jsonl,e5_flat_inner.index}).
CORPUS_PATH="${CORPUS_PATH:-${EVAL_CORPORA_DIR}/${DATASET}_${SPLIT}/corpus.jsonl}"
INDEX_PATH="${INDEX_PATH:-${EVAL_CORPORA_DIR}/${DATASET}_${SPLIT}/e5_Flat.index}"
[ -f "${CORPUS_PATH}" ] || { echo "Missing eval corpus: ${CORPUS_PATH} — run prepare_eval_corpus.py first"; exit 1; }
[ -f "${INDEX_PATH}" ]  || { echo "Missing eval index:  ${INDEX_PATH}  — run build_index_hotpotqa.py first"; exit 1; }

# Unique retrieval port per job. Deterministic from SLURM_JOB_ID so we can
# kill it cleanly on exit. Range 8000..8999.
RETRIEVAL_PORT="$((8000 + SLURM_JOB_ID % 1000))"
RETRIEVAL_URL="http://127.0.0.1:${RETRIEVAL_PORT}/retrieve"

CKPT_TAG="$(basename "${MERGED_CKPT}")"
OUTPUT_DIR="${PROJECT_DIR}/eval_results"
SAMPLING_TAG="t${TEMPERATURE}_tp${TOP_P}_tk${TOP_K_SAMPLE}_mt${MAX_TURNS}_resp${MAX_RESPONSE_LENGTH}"
THINK_TAG="$([ "${ENABLE_THINKING}" = "true" ] && echo "thinkT" || echo "thinkF")"
OUTPUT_PATH="${OUTPUT_DIR}/${CKPT_TAG}_${DATASET}_${SPLIT}_n${NUM_SAMPLES}_${PROMPT_VARIANT}_${THINK_TAG}_${SAMPLING_TAG}${TAG:+_${TAG}}_${SLURM_JOB_ID}.json"

echo "[$(date)] === a2a-v2 eval start ==="
echo "Node:                 $(hostname)"
echo "JobID:                ${SLURM_JOB_ID}"
echo "MERGED_CKPT:          ${MERGED_CKPT}"
echo "DATASET / SPLIT:      ${DATASET} / ${SPLIT}"
echo "PROMPT_VARIANT:       ${PROMPT_VARIANT}"
echo "ENABLE_THINKING:      ${ENABLE_THINKING}"
echo "NUM_SAMPLES:          ${NUM_SAMPLES}"
echo "CORPUS:               ${CORPUS_PATH}"
echo "INDEX:                ${INDEX_PATH}"
echo "RETRIEVAL_PORT:       ${RETRIEVAL_PORT}"
echo "OUTPUT:               ${OUTPUT_PATH}"
nvidia-smi -L

# ----- 1. Env -----
[ -x "${ENV_DIR}/bin/python" ] || { echo "mem env missing at ${ENV_DIR}"; exit 1; }
export PATH="${ENV_DIR}/bin:${PATH}"
python3 --version
python3 -c "import vllm; print('vllm', vllm.__version__)"

# Per-job vLLM/inductor cache. Use /scratch (large, node-local) not /tmp
# (only 15GB, often exhausted on aimi nodes).
export VLLM_CACHE_ROOT="/scratch/${USER}/vllm_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="/scratch/${USER}/inductor_${SLURM_JOB_ID}"
mkdir -p "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}"

# Pick a python for the retrieval server (needs faiss + e5)
if [ -x "/scratch/${USER}/envs/searchr1-sgl/bin/python" ]; then
    export ENV_PYTHON="/scratch/${USER}/envs/searchr1-sgl/bin/python"
else
    export ENV_PYTHON="${ENV_DIR}/bin/python"
fi

# ----- 2. Launch our own retrieval server (no sharing) -----
cd "${PROJECT_DIR}"
mkdir -p logs "${OUTPUT_DIR}"
RETRIEVAL_LOG="${PROJECT_DIR}/logs/retrieval_eval_a2a_${SLURM_JOB_ID}.log"

echo "[$(date)] Launching retrieval server on port ${RETRIEVAL_PORT}"
# Note: env vars must be BEFORE `setsid` so they're set on the subprocess
# environment, not passed as args to setsid itself.
( exec 200>&- 2>/dev/null || true
  CUDA_VISIBLE_DEVICES=0 \
  INDEX_PATH="${INDEX_PATH}" \
  CORPUS_PATH="${CORPUS_PATH}" \
  PORT="${RETRIEVAL_PORT}" \
      setsid bash launch_retrieval_verl.sh > "${RETRIEVAL_LOG}" 2>&1 < /dev/null &
  RETRIEVAL_PID=$!
  echo "${RETRIEVAL_PID}" > "/scratch/${USER}/retrieval_pid_${SLURM_JOB_ID}.txt"
  disown
)
RETRIEVAL_PID="$(cat /scratch/${USER}/retrieval_pid_${SLURM_JOB_ID}.txt 2>/dev/null)"
echo "[$(date)] retrieval pid: ${RETRIEVAL_PID}"

# Wait for /health. Full-wiki index (64GB) can take 5+ min to mmap from NFS,
# so make the timeout overridable. Small per-benchmark indices come up in <30s.
RETRIEVAL_HEALTH_TIMEOUT="${RETRIEVAL_HEALTH_TIMEOUT:-120}"
for i in $(seq 1 "${RETRIEVAL_HEALTH_TIMEOUT}"); do
    if curl -sf -m 2 "http://127.0.0.1:${RETRIEVAL_PORT}/health" > /dev/null 2>&1; then
        echo "[$(date)] retrieval healthy after ${i}s"; break
    fi
    sleep 1
done
curl -sf -m 5 "http://127.0.0.1:${RETRIEVAL_PORT}/health" > /dev/null \
    || { echo "retrieval never became healthy — abort"; tail -50 "${RETRIEVAL_LOG}"; exit 1; }

cleanup() {
    # Kill our own retrieval server. Other jobs use other ports.
    if [ -n "${RETRIEVAL_PID:-}" ]; then
        pkill -P "${RETRIEVAL_PID}" 2>/dev/null || true
        kill -- -"${RETRIEVAL_PID}" 2>/dev/null || true
        kill "${RETRIEVAL_PID}" 2>/dev/null || true
    fi
    # Also kill anything still bound to our port (belt-and-suspenders)
    fuser -k "${RETRIEVAL_PORT}/tcp" 2>/dev/null || true
    rm -f "/scratch/${USER}/retrieval_pid_${SLURM_JOB_ID}.txt" 2>/dev/null || true
    rm -rf "${VLLM_CACHE_ROOT}" "${TORCHINDUCTOR_CACHE_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

# ----- 3. Eval -----
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
    --prompt_variant "${PROMPT_VARIANT}" \
    --enable_thinking "${ENABLE_THINKING}" \
    --max_tool_response_length "${MAX_TOOL_RESPONSE_LENGTH:-512}" \
    --tool_response_truncate_side "${TOOL_RESPONSE_TRUNCATE_SIDE:-left}" \
    --retrieval_url "${RETRIEVAL_URL}" \
    --output_path "${OUTPUT_PATH}"

echo "[$(date)] === eval done — output: ${OUTPUT_PATH} ==="
python3 -c "
import json
with open('${OUTPUT_PATH}') as f:
    d = json.load(f)
print('SUMMARY:', json.dumps(d['summary'], indent=2))
"
