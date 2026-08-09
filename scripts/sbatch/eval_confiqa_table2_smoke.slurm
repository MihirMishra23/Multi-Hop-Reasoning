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
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${QUERY_SAMPLES}}"
SAVE_LABEL="${SAVE_LABEL:-smoke${QUERY_SAMPLES}_store${STORE_SAMPLES}}"
SEED="${SEED:-42}"
RUN_ROOT="${RUN_ROOT:-/share/j_sun/lmlm_multihop/results/confiqa_conflict_free_smoke}"
PYTHON="${PYTHON:-/home/rtn27/.conda/envs/multi-hop-reasoning/bin/python}"
RETRIEVER_PYTHON="${RETRIEVER_PYTHON:-/home/lz586/.conda/envs/sr1-retriever/bin/python}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR}}"
SR1_CODE_REV="${SR1_CODE_REV:-10a7b9b}"
export PATH="$(dirname "${PYTHON}"):${PATH}"
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-21-openjdk-amd64}"
export PATH="${JAVA_HOME}/bin:${PATH}"

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
        --batch-size "${EVAL_BATCH_SIZE}" \
        --save-every 1 \
        --output-dir "${RUN_ROOT}/kbevo" \
        --save-version "${SAVE_LABEL}" \
        --concat-all-db \
        --use-contexts all \
        --use-inverses \
        --top-k 4 \
        --similarity-threshold 0.6 \
        --use-train-params \
        --eval
elif [[ "${MODE}" == "searchr1" ]]; then
    MODEL_TAG="$(basename "${MODEL_PATH}")"
    WORK_DIR="${RUN_ROOT}/searchr1_recent_master/${MODEL_TAG}/${CONFIQA_CONDITION}_q${QUERY_SAMPLES}_store${STORE_SAMPLES}"
    BM25_INPUT="${WORK_DIR}/bm25_input_${SLURM_JOB_ID}"
    BM25_INDEX="${WORK_DIR}/bm25_index_${SLURM_JOB_ID}"
    SERVER_LOG="${WORK_DIR}/retrieval_server_${SLURM_JOB_ID}.log"
    mkdir -p "${WORK_DIR}" "${BM25_INPUT}"
    "${PYTHON}" rebuttal-search-r1/prepare_confiqa_corpus.py \
        --setting "${CONFIQA_CONDITION}" \
        --num-samples "${STORE_SAMPLES}" \
        --seed "${SEED}" \
        --output-dir "${WORK_DIR}"
    ln -s "${WORK_DIR}/corpus.jsonl" "${BM25_INPUT}/corpus.jsonl"
    "${RETRIEVER_PYTHON}" -m pyserini.index.lucene \
        --collection JsonCollection \
        --input "${BM25_INPUT}" \
        --index "${BM25_INDEX}" \
        --generator DefaultLuceneDocumentGenerator \
        --threads 8 \
        --storeRaw

    PORT=$((8000 + (SLURM_JOB_ID % 20000)))
    while ss -ltn 2>/dev/null | grep -q ":${PORT} "; do
        PORT=$((PORT + 1))
    done
    set +x
    export OPENAI_API_KEY=unused-by-bm25
    "${RETRIEVER_PYTHON}" Search-R1/search_r1/search/retrieval_server.py \
        --retriever_name bm25 \
        --index_path "${BM25_INDEX}" \
        --corpus_path "${WORK_DIR}/corpus.jsonl" \
        --topk 3 \
        --port "${PORT}" >"${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    set -x
    cleanup_search_server() {
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    }
    trap cleanup_search_server EXIT

    READY=0
    for _ in $(seq 1 120); do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "Search-R1 retrieval server exited during startup" >&2
            tail -n 100 "${SERVER_LOG}" >&2
            exit 1
        fi
        if curl -sf -o /dev/null -X POST "http://127.0.0.1:${PORT}/retrieve" \
            -H 'Content-Type: application/json' \
            -d '{"queries":["warmup"],"topk":1,"return_scores":true}'; then
            READY=1
            break
        fi
        sleep 5
    done
    if [[ "${READY}" -ne 1 ]]; then
        echo "Search-R1 retrieval server was not ready after 600 seconds" >&2
        tail -n 100 "${SERVER_LOG}" >&2
        exit 1
    fi

    "${PYTHON}" src/eval_multihop.py \
        --method search_r1 \
        --dataset confiqa \
        --split dev \
        --confiqa-setting "${CONFIQA_CONDITION}" \
        --model-path "${MODEL_PATH}" \
        --search-r1-retrieval-url "http://127.0.0.1:${PORT}/retrieve" \
        --search-r1-retrieval-k 3 \
        --search-r1-corpus-manifest "${WORK_DIR}/manifest.json" \
        --search-r1-code-revision "${SR1_CODE_REV}" \
        --max-steps 10 \
        --search-r1-max-response-length 10240 \
        --search-r1-max-model-len 16384 \
        --temperature 0.0 \
        --top-p 0.95 \
        --sampling-top-k -1 \
        --total-count "${QUERY_SAMPLES}" \
        --knowledge-store-count "${STORE_SAMPLES}" \
        --batch-size "${EVAL_BATCH_SIZE}" \
        --save-every 1 \
        --output-dir "${RUN_ROOT}/searchr1_recent_master" \
        --save-version "${SAVE_LABEL}_master_${SR1_CODE_REV}" \
        --seed "${SEED}" \
        --eval
else
    echo "Unsupported MODE=${MODE}" >&2
    exit 2
fi
