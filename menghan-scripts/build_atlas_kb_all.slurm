#!/bin/bash
# Build AutoSchemaKG (atlas-rag) KBs for selected datasets — 1000 dev questions,
# seed=42, same slice as KBevo paper Table 5.
#
# Each dataset gets one sbatch job. Each job:
#   1) build atlas-rag input JSON (1000q × ~10-20 passages)
#   2) start vllm serve Llama-3.1-8B-Instruct
#   3) atlas-rag KnowledgeGraphExtractor → 3-stage extraction
#   4) kill vllm
#   5) convert atlas-rag output → KBevo unified KB JSON
#
# Usage (login node):
#   # all 3 datasets on B200 (default)
#   bash menghan-scripts/build_atlas_kb_all.slurm
#
#   # specific dataset(s) on B200
#   bash menghan-scripts/build_atlas_kb_all.slurm hotpotqa
#   bash menghan-scripts/build_atlas_kb_all.slurm musique 2wiki
#
#   # on A6000 instead (about 3-5× slower; time budgets scaled up)
#   GPU=a6000 bash menghan-scripts/build_atlas_kb_all.slurm 2wiki

cd /share/j_sun/mx253/Multi-Hop-Reasoning

# ── Parse args ───────────────────────────────────────────────────────────────
if [[ $# -ge 1 ]]; then
    DATASETS=("$@")
else
    DATASETS=(hotpotqa musique 2wiki)
fi

# GPU type: b200 (default) or a6000
GPU="${GPU:-b200}"
case "${GPU}" in
    b200)
        GRES="gpu:nvidia_b200:1"
        VLLM_GPU_MEM=0.55
        # Time budgets with the 3-4x throughput tuning below (batch=64, workers=32).
        # Generous safety margin (2x of empirical estimate).
        TIME_BUDGET_HOTPOT="3:00:00"
        TIME_BUDGET_MUSIQUE="6:00:00"
        TIME_BUDGET_2WIKI="3:00:00"
        ;;
    a6000)
        GRES="gpu:nvidia_rtx_a6000:1"
        VLLM_GPU_MEM=0.85   # A6000 is 48GB; we own the GPU during build, can use more
        TIME_BUDGET_HOTPOT="12:00:00"
        TIME_BUDGET_MUSIQUE="24:00:00"
        TIME_BUDGET_2WIKI="12:00:00"
        ;;
    *)
        echo "Error: GPU must be 'b200' or 'a6000', got '${GPU}'" >&2
        exit 1
        ;;
esac

# ── Throughput tuning ────────────────────────────────────────────────────────
# Empirically the smoke test was severely underutilizing vLLM. With batch=64 +
# 32 concurrent workers + vLLM max_num_seqs=128 we expect ~3-4x speedup over
# the smoke config (batch=16, workers=8).
ATLAS_BATCH_SIZE=64
ATLAS_MAX_WORKERS=32
VLLM_MAX_NUM_SEQS=128
VLLM_MAX_BATCHED_TOKENS=16384

echo "GPU type: ${GPU}  (GRES=${GRES}, vLLM mem util=${VLLM_GPU_MEM})"
echo "Datasets to submit: ${DATASETS[*]}"
echo

# ── Fixed config ─────────────────────────────────────────────────────────────
N_QUESTIONS=1000
SEED=42
SPLIT=dev
SETTING=distractor

EXTRACTOR_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
VLLM_PORT=8765
VLLM_HOST="127.0.0.1"

KB_PKG=kb_baselines/autoschemakg
ATLAS_BASE_DIR="${KB_PKG}/output"

declare -A TIME_BUDGET=(
    [hotpotqa]="${TIME_BUDGET_HOTPOT}"
    [musique]="${TIME_BUDGET_MUSIQUE}"
    [2wiki]="${TIME_BUDGET_2WIKI}"
)

for DATASET in "${DATASETS[@]}"; do
    TAG="${DATASET}_${SPLIT}_n${N_QUESTIONS}_seed${SEED}"
    INPUT_JSON="${ATLAS_BASE_DIR}/inputs/${TAG}.json"
    ATLAS_OUTPUT_DIR="${ATLAS_BASE_DIR}/extractions/${TAG}"
    KBEVO_KB_JSON="${ATLAS_BASE_DIR}/kbs/${TAG}_kb.json"

    # Skip if KB already built
    if [[ -f "${KBEVO_KB_JSON}" ]]; then
        echo "Skipping ${DATASET}: KB already exists at ${KBEVO_KB_JSON}"
        continue
    fi

    JOB_NAME="atlas_kb_${DATASET}_${GPU}"
    TIME_BUDG="${TIME_BUDGET[$DATASET]}"
    TMP_SCRIPT=$(mktemp /tmp/atlas_kb_${DATASET}_XXXXXX.sh)

    cat > "${TMP_SCRIPT}" << SCRIPT_EOF
#!/bin/bash
set -euo pipefail
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate lmlm_b200
cd /share/j_sun/mx253/Multi-Hop-Reasoning

mkdir -p "$(dirname ${INPUT_JSON})" "${ATLAS_OUTPUT_DIR}" "$(dirname ${KBEVO_KB_JSON})"

echo "============================================================"
echo "[1/5] Build atlas-rag input JSON (${DATASET}, ${N_QUESTIONS} questions)"
echo "============================================================"
PYTHONPATH=src python ${KB_PKG}/build_atlas_input.py \\
    --dataset "${DATASET}" --split "${SPLIT}" --setting "${SETTING}" \\
    --n ${N_QUESTIONS} --seed ${SEED} \\
    --use-contexts all \\
    --output "${INPUT_JSON}"

echo "============================================================"
echo "[2/5] Launch vllm serve in background"
echo "============================================================"
VLLM_LOG="${ATLAS_OUTPUT_DIR}/vllm_serve.log"
nohup vllm serve "${EXTRACTOR_MODEL}" \\
    --host ${VLLM_HOST} --port ${VLLM_PORT} \\
    --gpu-memory-utilization ${VLLM_GPU_MEM} \\
    --max-model-len 8192 \\
    --max-num-seqs ${VLLM_MAX_NUM_SEQS} \\
    --max-num-batched-tokens ${VLLM_MAX_BATCHED_TOKENS} \\
    --dtype bfloat16 \\
    > "\${VLLM_LOG}" 2>&1 &
VLLM_PID=\$!
echo "vllm pid=\${VLLM_PID}  log=\${VLLM_LOG}"
trap "echo 'Cleaning up vllm pid=\${VLLM_PID}'; kill \${VLLM_PID} 2>/dev/null || true" EXIT

echo "Waiting for vllm to come up..."
for i in \$(seq 1 60); do
    if curl -s --max-time 2 "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        echo "vllm is ready after \${i}*10s"
        break
    fi
    sleep 10
    if [ \$i -eq 60 ]; then
        echo "vllm failed to come up in 10 minutes. Last 50 lines of log:"
        tail -50 "\${VLLM_LOG}"
        exit 1
    fi
done

echo "============================================================"
echo "[3/5] atlas-rag KG extraction (${DATASET}, 3 stages)"
echo "============================================================"
PYTHONPATH=src python ${KB_PKG}/run_atlas_extraction.py \\
    --input "${INPUT_JSON}" \\
    --output-dir "${ATLAS_OUTPUT_DIR}" \\
    --vllm-endpoint "http://${VLLM_HOST}:${VLLM_PORT}/v1" \\
    --model-name "${EXTRACTOR_MODEL}" \\
    --batch-size ${ATLAS_BATCH_SIZE} \\
    --max-workers ${ATLAS_MAX_WORKERS}

echo "============================================================"
echo "[4/5] Kill vllm server"
echo "============================================================"
kill \${VLLM_PID} 2>/dev/null || true
wait \${VLLM_PID} 2>/dev/null || true
trap - EXIT
sleep 5

echo "============================================================"
echo "[5/5] Convert atlas-rag output -> KBevo unified KB JSON"
echo "============================================================"
PYTHONPATH=src python ${KB_PKG}/atlas_to_kbevo_kb.py \\
    --atlas-output-dir "${ATLAS_OUTPUT_DIR}" \\
    --output "${KBEVO_KB_JSON}"

echo
echo "==== DONE: ${DATASET} ===="
echo "KB: ${KBEVO_KB_JSON}"
SCRIPT_EOF

    echo "Submitting: ${JOB_NAME} (GPU=${GPU}, time=${TIME_BUDG})"
    sbatch \
        -J "${JOB_NAME}" \
        -o "/home/mx253/slurm_output/lmlm_multihop/${JOB_NAME}_%j.out" \
        -e "/home/mx253/slurm_output/lmlm_multihop/${JOB_NAME}_%j.err" \
        -N 1 -n 1 \
        --cpus-per-task=8 \
        --mem=128G \
        -t "${TIME_BUDG}" \
        --partition=aimi,default_partition \
        --exclude=jjs533-compute-01,jjs533-compute-02,jjs533-compute-03 \
        --gres="${GRES}" \
        --requeue \
        "${TMP_SCRIPT}"
    rm -f "${TMP_SCRIPT}"
done

echo
echo "All requested KB build jobs submitted. Status:"
echo "  squeue -u \$USER -o '%.10i %.30j %.4t %.10M %R'"
