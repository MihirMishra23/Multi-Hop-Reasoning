#!/bin/bash
#
# Search-R1 evaluation through the shared evaluator (src/eval_multihop.py,
# --method search_r1, added by PR #46).
#
# This is a DISPATCHER, not a job: run it with `bash`, and it submits one job
# per (model, dataset) pair. Each submitted job is self-contained — it starts
# its own BM25 retrieval server on a private port, waits for it to answer,
# runs the eval, then shuts the server down.
#
# Two conda envs are used on purpose. pyserini pulls torch 2.13 + CUDA 13,
# which would break vLLM/DeepSpeed in `mem`, so the retriever lives in its own
# env and talks to the evaluator over HTTP:
#
#     retrieval server  ->  RETRIEVER_ENV  (pyserini, fastapi, uvicorn)
#     eval_multihop.py  ->  EVAL_ENV       (vllm, transformers)
#
# Usage:
#     bash scripts/lx_scripts/eval_search_r1.slurm
#
#SBATCH -J search_r1_dispatch
#SBATCH -o /home/lz586/icl/slurm_output/lmlm_multi_hop/search_r1_dispatch_%j.out
#SBATCH -e /home/lz586/icl/slurm_output/lmlm_multi_hop/search_r1_dispatch_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH --partition=kilian,default_partition

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

REPO=/share/j_sun/mx253_lmlm
DATA_ROOT=/share/j_sun/mx253/Search-R1     # index/, corpus/ live here (owned by mx253, read-only)

EVAL_ENV=mem                                # vllm + transformers
RETRIEVER_ENV=sr1-retriever                 # pyserini + fastapi + uvicorn

SAVE_VERSION="_0806_sr1_n1000_3b_noshuf"
OUTPUT_DIR="./output/main_tables"

# Set to a small number (e.g. 20) for a smoke test before committing to full dev.
# 20000 auto-caps to each dataset's dev size (hotpotqa 7405 / musique 2417 / 2wiki 12576).
NUM_SAMPLES=1000

# Search-R1 retrieval / rollout parameters, matched to upstream
# evaluate_searchr1.py so results are comparable to the published numbers:
#   while cnt < 10          -> MAX_STEPS=10
#   max_new_tokens=1024     -> 10 turns x 1024 = 10240 total budget
#   do_sample=False, T=0.0  -> greedy (already the evaluator default)
#   samples[:max_samples]   -> NO_SHUFFLE, upstream takes the first N unshuffled
# Upstream sets no context cap (HF generate uses the model's native window:
# 32768 for 3B, 131072 for 7B). 16384 is well above what 10 turns of 3 docs
# needs while keeping the KV cache affordable on a 48GB card.
RETRIEVAL_TOPK=3
PROMPT_VARIANT="default"                    # default | toolcall | thinkingtag | icl3hop
NO_SHUFFLE="--no-shuffle"                   # "" to shuffle by --seed instead
MAX_STEPS=10
MAX_RESPONSE_LENGTH=10240
SEARCH_R1_MAX_MODEL_LEN=16384

# Per-job SLURM resources
EVAL_PARTITION="kilian,default_partition"
EVAL_GRES="gpu:nvidia_rtx_a6000:1"
EVAL_CPUS=8
EVAL_MEM=128G

# ── Models ────────────────────────────────────────────────────────────────────
# v0.1 checkpoints (outcome-only RL, paper 2503.09516), base — not -it — start.
# NOTE: mx253's existing results are mostly v0.3, so numbers from here are NOT
#       directly comparable to those; this is a deliberate v0.1 run.
#   short_name : hf_repo_id_or_local_path
MODEL_LST=(
    "sr1-3b:PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo"
    # "sr1-7b:PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"
)

# ── Datasets ──────────────────────────────────────────────────────────────────
# Verified full dev sets (count matches official dev size; corpus answer
# coverage 100% on a 400-question sample vs ~45% cross-dataset control).
# popqa is intentionally absent: only 1000/423-question subsets exist.
#
#   name : corpus jsonl : bm25 index dir
DATASETS=(
    "hotpotqa:hotpotqa_distractor_full_val_corpus.jsonl:hotpotqa_full_val"
    # musique excluded from the unshuffled comparison: our loader reads it from
    # HF and mx253 from an exported json, so the orders differ (2/1000 match).
    # "musique:musique_val_corpus.jsonl:musique_val"
    "2wiki:2wiki_val_corpus.jsonl:2wiki_val"
)

# ── Preflight ─────────────────────────────────────────────────────────────────

CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
if [[ ! -f "${CONDA_SH}" ]]; then
    echo "ERROR: cannot locate conda.sh" >&2
    exit 1
fi
# Envs may live outside $(conda info --base)/envs (here they are under
# ~/.conda/envs), so ask conda rather than guessing the path.
for ENV_NAME in "${RETRIEVER_ENV}" "${EVAL_ENV}"; do
    if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        echo "ERROR: conda env '${ENV_NAME}' not found." >&2
        [[ "${ENV_NAME}" == "${RETRIEVER_ENV}" ]] && {
            echo "  conda create -n ${RETRIEVER_ENV} python=3.10 -y" >&2
            echo "  conda activate ${RETRIEVER_ENV} && pip install pyserini fastapi uvicorn" >&2
        }
        exit 1
    fi
done

# retrieval_server.py must accept --port (local patch). Without it every job
# would bind 8000 and collide on shared nodes.
if ! grep -qE "add_argument\((['\"])--port\1" "${REPO}/Search-R1/search_r1/search/retrieval_server.py"; then
    echo "ERROR: retrieval_server.py has no --port option; apply the local patch first." >&2
    exit 1
fi

# ── Submit one job per (model, dataset) ───────────────────────────────────────

N_SUBMITTED=0
for MENTRY in "${MODEL_LST[@]}"; do
    MODEL_TAG="${MENTRY%%:*}"
    MODEL_PATH="${MENTRY#*:}"

    # Accept either a local directory or a HuggingFace repo id (org/name).
    if [[ ! -d "${MODEL_PATH}" && ! "${MODEL_PATH}" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then
        echo "WARN: skipping ${MODEL_TAG}: not a local dir and not a repo id: ${MODEL_PATH}" >&2
        continue
    fi

    for DENTRY in "${DATASETS[@]}"; do
        IFS=':' read -r DS CORPUS INDEX <<< "${DENTRY}"

        INDEX_PATH="${DATA_ROOT}/index/${INDEX}/bm25"
        CORPUS_PATH="${DATA_ROOT}/corpus/${CORPUS}"

        if [[ ! -d "${INDEX_PATH}" ]]; then
            echo "WARN: missing index, skipping ${MODEL_TAG}/${DS}: ${INDEX_PATH}" >&2
            continue
        fi
        if [[ ! -f "${CORPUS_PATH}" ]]; then
            echo "WARN: missing corpus, skipping ${MODEL_TAG}/${DS}: ${CORPUS_PATH}" >&2
            continue
        fi

        TMP_SCRIPT=$(mktemp /tmp/search_r1_eval_XXXXXX.sh)
        cat > "${TMP_SCRIPT}" << SCRIPT_EOF
#!/bin/bash
# NOTE: deliberately no 'set -u'. The mem env's conda activate hook
# (activate.d/libblas_mkl_activate.sh) reads an unset MKL_INTERFACE_LAYER,
# which under 'set -u' aborts the script the moment we activate it.
set -o pipefail
source ${CONDA_SH}

# Derive a per-job port so several eval jobs can share a node, then step past
# anything already bound. Requires the --port patch in retrieval_server.py.
PORT=\$(( 8000 + (SLURM_JOB_ID % 20000) ))
while ss -ltn 2>/dev/null | grep -q ":\${PORT} "; do
    PORT=\$(( PORT + 1 ))
done
echo "[\$(date +%T)] node=\$(hostname) port=\${PORT} model=${MODEL_TAG} dataset=${DS}"

conda activate ${RETRIEVER_ENV}
# pyserini 1.2.0 instantiates an OpenAI client at import time in
# pyserini/encode/_openai.py, and LuceneSearcher's import chain goes through
# it, so importing it without a key raises OpenAIError. BM25 never calls the
# encoder — this placeholder only gets the import through, no request is made.
export OPENAI_API_KEY=\${OPENAI_API_KEY:-unused-by-bm25}
cd ${REPO}/Search-R1
python search_r1/search/retrieval_server.py \\
    --retriever_name bm25 \\
    --index_path  ${INDEX_PATH} \\
    --corpus_path ${CORPUS_PATH} \\
    --topk ${RETRIEVAL_TOPK} \\
    --port \${PORT} &
SERVER_PID=\$!

# Always take the server down, however this script exits.
cleanup() {
    echo "[\$(date +%T)] stopping retrieval server (pid \$SERVER_PID)"
    kill \$SERVER_PID 2>/dev/null || true
    wait \$SERVER_PID 2>/dev/null || true
}
trap cleanup EXIT

# Wait for a real query to succeed. Bail out after ~10 min instead of holding a
# GPU for hours if the server died while loading the index.
echo "[\$(date +%T)] waiting for retrieval server"
READY=0
for i in \$(seq 1 120); do
    if ! kill -0 \$SERVER_PID 2>/dev/null; then
        echo "FATAL: retrieval server died during startup" >&2
        exit 1
    fi
    if curl -sf -o /dev/null -X POST http://127.0.0.1:\${PORT}/retrieve \\
         -H 'Content-Type: application/json' \\
         -d '{"queries":["warmup"],"topk":1,"return_scores":true}'; then
        READY=1
        echo "[\$(date +%T)] retrieval server ready after \$((i*5))s"
        break
    fi
    sleep 5
done
if [[ \$READY -ne 1 ]]; then
    echo "FATAL: retrieval server not ready after 600s" >&2
    exit 1
fi

echo "[\$(date +%T)] starting eval: ${MODEL_TAG} / ${DS}"
conda activate ${EVAL_ENV}
cd ${REPO}
python src/eval_multihop.py \\
    --method search_r1 \\
    --dataset ${DS} \\
    --split dev \\
    --setting distractor \\
    --model-path "${MODEL_PATH}" \\
    --search-r1-retrieval-url http://127.0.0.1:\${PORT}/retrieve \\
    --search-r1-retrieval-k ${RETRIEVAL_TOPK} \\
    --search-r1-prompt-variant ${PROMPT_VARIANT} \\
    ${NO_SHUFFLE} \\
    --max-steps ${MAX_STEPS} \\
    --search-r1-max-response-length ${MAX_RESPONSE_LENGTH} \\
    --search-r1-max-model-len ${SEARCH_R1_MAX_MODEL_LEN} \\
    --total-count ${NUM_SAMPLES} \\
    --output-dir "${OUTPUT_DIR}/" \\
    --save-version "${SAVE_VERSION}" \\
    --seed 42 \\
    --eval \\
    --resume
STATUS=\$?
echo "[\$(date +%T)] eval finished: ${MODEL_TAG} / ${DS} status=\$STATUS"
exit \$STATUS
SCRIPT_EOF

        chmod +x "${TMP_SCRIPT}"

        JOB_NAME="sr1_eval_${DS}_${MODEL_TAG}"
        echo "Submitting: ${JOB_NAME}"
        sbatch \
            --job-name="${JOB_NAME}" \
            --output="/home/lz586/icl/slurm_output/lmlm_multi_hop/${JOB_NAME}_%j.out" \
            --error="/home/lz586/icl/slurm_output/lmlm_multi_hop/${JOB_NAME}_%j.err" \
            --partition="${EVAL_PARTITION}" \
            --gres="${EVAL_GRES}" \
            --cpus-per-task=${EVAL_CPUS} \
            --mem=${EVAL_MEM} \
            --time=infinite \
            --requeue \
            "${TMP_SCRIPT}"
        N_SUBMITTED=$(( N_SUBMITTED + 1 ))
    done
done

echo "Submitted ${N_SUBMITTED} Search-R1 eval jobs."
