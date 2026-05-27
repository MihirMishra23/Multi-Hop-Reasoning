#!/bin/bash
# Experiment B Step 1: Build atlas-rag KB on the SAME training slice as the
# Gemini Phase-2-only training (Table 5 "Train w/ Phase-2 Only" row).
#
# Slice: hotpotqa train, seed=42, START_IDX=82347, N=8100
#        (= last 8100 of seeded-shuffled 90447-example train split)
# Matches: /share/j_sun/lmlm_multihop/database/gemini/hotpotqa_train_start_idx_82347_nb_8100_database.json
#
# Output: kb_baselines/autoschemakg/output/kbs/hotpotqa_train_82347_8100_seed42_kb.json
#
# Runtime estimate (with our throughput tuning):
#   - 8100 questions × ~10 passages = ~81000 passages
#   - On B200: ~3-4h
#   - On A6000: ~10-12h
#
# Usage:
#   bash menghan-scripts/build_atlas_kb_train.slurm           # default B200
#   GPU=a6000 bash menghan-scripts/build_atlas_kb_train.slurm

cd /share/j_sun/mx253/Multi-Hop-Reasoning

GPU="${GPU:-b200}"
case "${GPU}" in
    b200)
        GRES="gpu:nvidia_b200:1"
        VLLM_GPU_MEM=0.55
        TIME_BUDG="6:00:00"
        ;;
    a6000)
        GRES="gpu:nvidia_rtx_a6000:1"
        VLLM_GPU_MEM=0.85
        TIME_BUDG="18:00:00"
        ;;
    *)
        echo "Error: GPU must be 'b200' or 'a6000'" >&2; exit 1 ;;
esac

# ── Fixed slice (matches Gemini Phase-2-only training KB) ──
DATASET=hotpotqa
SPLIT=train
SETTING=distractor
SEED=42
START_IDX=82347
N=8100

EXTRACTOR_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
VLLM_PORT=8765
VLLM_HOST="127.0.0.1"
ATLAS_BATCH_SIZE=64
ATLAS_MAX_WORKERS=32
VLLM_MAX_NUM_SEQS=128
VLLM_MAX_BATCHED_TOKENS=16384

KB_PKG=kb_baselines/autoschemakg
ATLAS_BASE_DIR="${KB_PKG}/output"
TAG="${DATASET}_train_${START_IDX}_${N}_seed${SEED}"
INPUT_JSON="${ATLAS_BASE_DIR}/inputs/${TAG}.json"
ATLAS_OUTPUT_DIR="${ATLAS_BASE_DIR}/extractions/${TAG}"
KBEVO_KB_JSON="${ATLAS_BASE_DIR}/kbs/${TAG}_kb.json"

if [[ -f "${KBEVO_KB_JSON}" ]]; then
    echo "KB already exists: ${KBEVO_KB_JSON}"
    echo "Skipping. Delete the file to rebuild."
    exit 0
fi

JOB_NAME="atlas_kb_train_${DATASET}_${GPU}"
TMP_SCRIPT=$(mktemp /tmp/atlas_kb_train_XXXXXX.sh)

cat > "${TMP_SCRIPT}" << SCRIPT_EOF
#!/bin/bash
set -euo pipefail
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate lmlm_b200
cd /share/j_sun/mx253/Multi-Hop-Reasoning

mkdir -p "$(dirname ${INPUT_JSON})" "${ATLAS_OUTPUT_DIR}" "$(dirname ${KBEVO_KB_JSON})"

echo "============================================================"
echo "[1/5] Build atlas-rag input JSON (${DATASET} TRAIN, ${N} questions, start=${START_IDX})"
echo "============================================================"
PYTHONPATH=src python ${KB_PKG}/build_atlas_input.py \\
    --dataset "${DATASET}" --split "${SPLIT}" --setting "${SETTING}" \\
    --n ${N} --seed ${SEED} --start-index ${START_IDX} \\
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
        echo "vllm is ready after \${i}*10s"; break
    fi
    sleep 10
    if [ \$i -eq 60 ]; then
        echo "vllm failed to come up in 10 minutes."
        tail -50 "\${VLLM_LOG}"; exit 1
    fi
done

echo "============================================================"
echo "[3/5] atlas-rag KG extraction (${DATASET} TRAIN, 3 stages)"
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
echo "==== DONE: ${DATASET} TRAIN ===="
echo "KB: ${KBEVO_KB_JSON}"
echo "Next: bash menghan-scripts/train_phase2_only_atlas.slurm"
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

echo
echo "Status: squeue -u \$USER -n ${JOB_NAME}"
