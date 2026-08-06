#!/usr/bin/env bash
# Prepare HotpotQA, build retrieval, and launch Linxi's paper training recipe.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-python3}"
MODEL_SIZE="${MODEL_SIZE:-1.7B}"
PORT="${PORT:-8000}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-$(
    "${PYTHON}" "${SCRIPT_DIR}/reproduction_manifest.py" \
        --artifact e5_base_v2 --field repo_id
)}"
RETRIEVER_REVISION="${RETRIEVER_REVISION:-$(
    "${PYTHON}" "${SCRIPT_DIR}/reproduction_manifest.py" \
        --artifact e5_base_v2 --field revision
)}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
INDEX_DIR="${INDEX_DIR:-${DATA_DIR}/index}"
RETRIEVAL_URL="${RETRIEVAL_URL:-http://127.0.0.1:${PORT}/retrieve}"
RETRIEVAL_HEALTH_URL="${RETRIEVAL_HEALTH_URL:-http://127.0.0.1:${PORT}/health}"
RETRIEVAL_LOG="${RETRIEVAL_LOG:-${SCRIPT_DIR}/logs/retrieval.log}"

mkdir -p "${DATA_DIR}" "${INDEX_DIR}" "$(dirname "${RETRIEVAL_LOG}")"

prepared_data=0
if [ "${FORCE_PREPARE:-0}" = "1" ] || \
   [ ! -f "${DATA_DIR}/train_verl.parquet" ] || \
   [ ! -f "${DATA_DIR}/test_verl.parquet" ] || \
   [ ! -f "${DATA_DIR}/hotpotqa_corpus.jsonl" ]; then
    "${PYTHON}" "${SCRIPT_DIR}/prepare_hotpotqa_data.py" --data-dir "${DATA_DIR}"
    prepared_data=1
fi

if [ "${FORCE_INDEX:-0}" = "1" ] || [ "${prepared_data}" = "1" ] || \
   [ ! -f "${INDEX_DIR}/e5_Flat.index" ] || \
   [ ! -f "${INDEX_DIR}/e5_Flat.index.manifest.json" ]; then
    "${PYTHON}" "${SCRIPT_DIR}/build_index_hotpotqa.py" \
        --corpus-path "${DATA_DIR}/hotpotqa_corpus.jsonl" \
        --output-dir "${INDEX_DIR}" \
        --model "${RETRIEVER_MODEL}" \
        --revision "${RETRIEVER_REVISION}" \
        --device "${INDEX_DEVICE:-cuda}"
fi

retrieval_pid=""
cleanup() {
    if [ -n "${retrieval_pid}" ]; then
        kill "${retrieval_pid}" 2>/dev/null || true
        wait "${retrieval_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if curl -fsS --max-time 2 "${RETRIEVAL_HEALTH_URL}" >/dev/null 2>&1; then
    [ "${REUSE_RETRIEVER:-0}" = "1" ] || {
        echo "A retriever is already running at ${RETRIEVAL_HEALTH_URL}." >&2
        echo "Stop it or set REUSE_RETRIEVER=1 explicitly." >&2
        exit 1
    }
else
    CUDA_VISIBLE_DEVICES="${RETRIEVAL_CUDA_VISIBLE_DEVICES:-0}" \
    PYTHON="${PYTHON}" \
    INDEX_PATH="${INDEX_DIR}/e5_Flat.index" \
    CORPUS_PATH="${DATA_DIR}/hotpotqa_corpus.jsonl" \
    RETRIEVER_MODEL="${RETRIEVER_MODEL}" \
    RETRIEVER_REVISION="${RETRIEVER_REVISION}" \
    RETRIEVER_DEVICE="${RETRIEVER_DEVICE:-cuda}" \
    PORT="${PORT}" \
        "${SCRIPT_DIR}/launch_retrieval_verl.sh" >"${RETRIEVAL_LOG}" 2>&1 &
    retrieval_pid=$!

    for _ in $(seq 1 "${RETRIEVAL_STARTUP_TIMEOUT:-180}"); do
        curl -fsS --max-time 2 "${RETRIEVAL_HEALTH_URL}" >/dev/null 2>&1 && break
        kill -0 "${retrieval_pid}" 2>/dev/null || {
            tail -50 "${RETRIEVAL_LOG}" >&2
            exit 1
        }
        sleep 1
    done
    curl -fsS --max-time 2 "${RETRIEVAL_HEALTH_URL}" >/dev/null || {
        echo "Retriever did not become healthy; see ${RETRIEVAL_LOG}" >&2
        exit 1
    }
fi

case "${MODEL_SIZE}" in
    1.7B|1.7b) launcher="${SCRIPT_DIR}/run_qwen3_1.7b_apples_to_apples.sh" ;;
    4B|4b) launcher="${SCRIPT_DIR}/run_qwen3_4b_apples_to_apples.sh" ;;
    *) echo "MODEL_SIZE must be 1.7B or 4B" >&2; exit 2 ;;
esac

export PYTHON
export TRAIN_DATA="${DATA_DIR}/train_verl.parquet"
export VAL_DATA="${DATA_DIR}/test_verl.parquet"
export RETRIEVAL_URL RETRIEVAL_HEALTH_URL

"${launcher}" "$@"
