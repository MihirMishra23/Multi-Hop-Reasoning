#!/bin/bash
#
# Launch our local dense retrieval server in the verl /retrieve protocol.
# FAISS stays on CPU (faiss-gpu-cu12 has no Blackwell kernels). E5 encoder
# uses GPU 0 by default; override with CUDA_VISIBLE_DEVICES.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INDEX_PATH="${INDEX_PATH:-${SCRIPT_DIR}/data/index/e5_Flat.index}"
CORPUS_PATH="${CORPUS_PATH:-${SCRIPT_DIR}/data/hotpotqa_corpus.jsonl}"
PORT="${PORT:-8000}"
ENV_PYTHON="${ENV_PYTHON:-/scratch/lz586/envs/searchr1-sgl/bin/python}"
WORKERS="${WORKERS:-1}"

[ -f "${INDEX_PATH}" ]  || { echo "Missing index: ${INDEX_PATH}"; exit 1; }
[ -f "${CORPUS_PATH}" ] || { echo "Missing corpus: ${CORPUS_PATH}"; exit 1; }
[ -x "${ENV_PYTHON}" ]  || { echo "Missing python: ${ENV_PYTHON}"; exit 1; }

echo "Index  : ${INDEX_PATH}"
echo "Corpus : ${CORPUS_PATH}"
echo "Port   : ${PORT}"
echo "Workers: ${WORKERS}"
echo "Python : ${ENV_PYTHON}"

exec "${ENV_PYTHON}" "${SCRIPT_DIR}/retrieval_server_verl.py" \
    --index_path "${INDEX_PATH}" \
    --corpus_path "${CORPUS_PATH}" \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --workers "${WORKERS}" \
    --port "${PORT}"
