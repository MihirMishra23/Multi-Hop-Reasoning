#!/bin/bash
#
# Reproduce the reference Search-R1 numbers by running mx253's own harness
# (evaluate_searchr1.py) rather than our agent, on the same index, corpus and
# model. This is the control for the discrepancy where our 3B scores ~8-10
# points above the reference while our 7B matches it.
#
# The only change to upstream's script is that retrieval goes over HTTP
# (scripts/lx_scripts/upstream/evaluate_searchr1_http.py) so pyserini can live
# in its own env; the BM25 index, top_k and prompt are untouched.
#
# Everything else is upstream's: HF generate (not vLLM), first N unshuffled
# examples, its own EM scoring.
#
# Usage:
#     bash scripts/lx_scripts/eval_search_r1_upstream.slurm
#
#SBATCH -J sr1_upstream_dispatch
#SBATCH -o /home/lz586/icl/slurm_output/lmlm_multi_hop/sr1_upstream_dispatch_%j.out
#SBATCH -e /home/lz586/icl/slurm_output/lmlm_multi_hop/sr1_upstream_dispatch_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH --partition=kilian,default_partition

set -o pipefail

REPO=/share/j_sun/mx253_lmlm
DATA_ROOT=/share/j_sun/mx253/Search-R1
UPSTREAM_PY=${REPO}/scripts/lx_scripts/upstream/evaluate_searchr1_http.py

EVAL_ENV=mem                 # torch + transformers (GPU)
RETRIEVER_ENV=sr1-retriever  # pyserini + fastapi + uvicorn

MAX_SAMPLES=1000             # upstream takes the first N unshuffled
OUT_ROOT=${REPO}/output/search_r1_upstream

# 3B only: our 7B already reproduces the reference within +-2.3 points.
MODEL_TAG="3b"

#   name : val json : corpus jsonl : bm25 index dir
DATASETS=(
    "hotpotqa:hotpotqa_val.json:hotpotqa_distractor_full_val_corpus.jsonl:hotpotqa_full_val"
    "2wiki:2wiki_val.json:2wiki_val_corpus.jsonl:2wiki_val"
)

CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
[[ -f "${CONDA_SH}" ]] || { echo "ERROR: no conda.sh" >&2; exit 1; }
[[ -f "${UPSTREAM_PY}" ]] || { echo "ERROR: missing ${UPSTREAM_PY}" >&2; exit 1; }

mkdir -p "${OUT_ROOT}"

for ENTRY in "${DATASETS[@]}"; do
    IFS=':' read -r DS VAL_JSON CORPUS INDEX <<< "${ENTRY}"

    VAL_PATH="${DATA_ROOT}/val_dataset/${VAL_JSON}"
    INDEX_PATH="${DATA_ROOT}/index/${INDEX}/bm25"
    CORPUS_PATH="${DATA_ROOT}/corpus/${CORPUS}"

    for p in "${VAL_PATH}" "${CORPUS_PATH}"; do
        [[ -f "$p" ]] || { echo "WARN: missing $p, skipping ${DS}" >&2; continue 2; }
    done
    [[ -d "${INDEX_PATH}" ]] || { echo "WARN: missing ${INDEX_PATH}, skipping ${DS}" >&2; continue; }

    TMP_SCRIPT=$(mktemp /tmp/sr1_upstream_XXXXXX.sh)
    cat > "${TMP_SCRIPT}" << SCRIPT_EOF
#!/bin/bash
# No 'set -u': mem's conda activate hook reads an unset MKL_INTERFACE_LAYER.
set -o pipefail
source ${CONDA_SH}

PORT=\$(( 8000 + (SLURM_JOB_ID % 20000) ))
while ss -ltn 2>/dev/null | grep -q ":\${PORT} "; do
    PORT=\$(( PORT + 1 ))
done
echo "[\$(date +%T)] node=\$(hostname) port=\${PORT} dataset=${DS} (upstream harness)"

conda activate ${RETRIEVER_ENV}
# pyserini 1.2.0 builds an OpenAI client at import time; BM25 never calls it.
export OPENAI_API_KEY=\${OPENAI_API_KEY:-unused-by-bm25}
cd ${REPO}/Search-R1
python search_r1/search/retrieval_server.py \\
    --retriever_name bm25 \\
    --index_path  ${INDEX_PATH} \\
    --corpus_path ${CORPUS_PATH} \\
    --topk 3 \\
    --port \${PORT} &
SERVER_PID=\$!
cleanup() { kill \$SERVER_PID 2>/dev/null || true; wait \$SERVER_PID 2>/dev/null || true; }
trap cleanup EXIT

READY=0
for i in \$(seq 1 120); do
    kill -0 \$SERVER_PID 2>/dev/null || { echo "FATAL: server died" >&2; exit 1; }
    if curl -sf -o /dev/null -X POST http://127.0.0.1:\${PORT}/retrieve \\
         -H 'Content-Type: application/json' \\
         -d '{"queries":["warmup"],"topk":1,"return_scores":true}'; then
        READY=1; echo "[\$(date +%T)] server ready after \$((i*5))s"; break
    fi
    sleep 5
done
[[ \$READY -eq 1 ]] || { echo "FATAL: server not ready" >&2; exit 1; }

echo "[\$(date +%T)] running upstream harness on ${DS}"
conda activate ${EVAL_ENV}
export RETRIEVAL_URL=http://127.0.0.1:\${PORT}/retrieve
cd ${REPO}
python ${UPSTREAM_PY} \\
    --data_path   ${VAL_PATH} \\
    --index_path  ${INDEX_PATH} \\
    --corpus_path ${CORPUS_PATH} \\
    --max_samples ${MAX_SAMPLES} \\
    --output_dir  ${OUT_ROOT}/${DS}_${MODEL_TAG}
STATUS=\$?
echo "[\$(date +%T)] done ${DS} status=\$STATUS"
exit \$STATUS
SCRIPT_EOF

    chmod +x "${TMP_SCRIPT}"
    JOB_NAME="sr1_upstream_${DS}_${MODEL_TAG}"
    echo "Submitting: ${JOB_NAME}"
    sbatch \
        --job-name="${JOB_NAME}" \
        --output="/home/lz586/icl/slurm_output/lmlm_multi_hop/${JOB_NAME}_%j.out" \
        --error="/home/lz586/icl/slurm_output/lmlm_multi_hop/${JOB_NAME}_%j.err" \
        --partition="kilian,default_partition" \
        --gres="gpu:nvidia_rtx_a6000:1" \
        --cpus-per-task=8 \
        --mem=128G \
        --time=infinite \
        --requeue \
        "${TMP_SCRIPT}"
done

echo "Submitted upstream-harness control jobs."
