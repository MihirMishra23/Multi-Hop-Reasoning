#!/usr/bin/env bash
# Launch the local retriever used by the Search-R1 training loop.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON="${PYTHON:-python3}"
INDEX_PATH="${INDEX_PATH:-${SCRIPT_DIR}/data/index/e5_Flat.index}"
CORPUS_PATH="${CORPUS_PATH:-${SCRIPT_DIR}/data/hotpotqa_corpus.jsonl}"
DEFAULT_RETRIEVER_MODEL="$(
    "${PYTHON}" "${SCRIPT_DIR}/reproduction_manifest.py" \
        --artifact e5_base_v2 --field repo_id
)"
DEFAULT_RETRIEVER_REVISION="$(
    "${PYTHON}" "${SCRIPT_DIR}/reproduction_manifest.py" \
        --artifact e5_base_v2 --field revision
)"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-${DEFAULT_RETRIEVER_MODEL}}"
RETRIEVER_REVISION="${RETRIEVER_REVISION:-${DEFAULT_RETRIEVER_REVISION}}"
RETRIEVER_DEVICE="${RETRIEVER_DEVICE:-cuda}"
PORT="${PORT:-8000}"

for path in "${INDEX_PATH}" "${CORPUS_PATH}"; do
    [ -f "${path}" ] || { echo "Missing retrieval artifact: ${path}" >&2; exit 1; }
done

exec "${PYTHON}" "${SCRIPT_DIR}/retrieval_server_verl.py" \
    --index-path "${INDEX_PATH}" \
    --corpus-path "${CORPUS_PATH}" \
    --model "${RETRIEVER_MODEL}" \
    --revision "${RETRIEVER_REVISION}" \
    --device "${RETRIEVER_DEVICE}" \
    --topk 3 \
    --port "${PORT}"
