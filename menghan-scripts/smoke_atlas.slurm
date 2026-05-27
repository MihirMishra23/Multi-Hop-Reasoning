#!/bin/bash
# W2 rebuttal SMOKE TEST: AutoSchemaKG static-KG baseline.
#
# 5 HotpotQA dev questions, end-to-end:
#   1) build atlas-rag input JSON (5 q × 10 paras = 50 passages)
#   2) start vllm serve Llama-3.1-8B-Instruct in background
#   3) atlas-rag KnowledgeGraphExtractor → 3-stage extraction via vLLM OpenAI API
#   4) kill vllm server (free GPU for the Phase-2 reasoner)
#   5) convert atlas-rag output → KBevo unified KB JSON
#   6) eval_multihop.py --edc-kb-path → Phase-2 reasoner with KBEVO-GRPO 1.7B
#
# Expected runtime on 1× A6000: ~25-40 min total
#   - vLLM warmup:    3-5 min
#   - extraction:     5-10 min for 50 passages × 3 stages
#   - reasoner eval:  10-15 min for 5 questions
#
# Run:
#   sbatch menghan-scripts/smoke_atlas.slurm

#SBATCH --job-name=smoke-atlas
#SBATCH --partition=aimi,default_partition
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --exclude=jjs533-compute-01,jjs533-compute-02,jjs533-compute-03
#SBATCH --time=2:00:00
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/smoke_atlas_%j.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/smoke_atlas_%j.err

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate lmlm_b200
cd /share/j_sun/mx253/Multi-Hop-Reasoning

# ── Configuration ────────────────────────────────────────────────────────────
N_QUESTIONS=5
SEED=42
DATASET=hotpotqa
SPLIT=dev
SETTING=distractor

EXTRACTOR_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
VLLM_PORT=8765
VLLM_HOST="127.0.0.1"
VLLM_GPU_MEM=0.55  # leaves room for the Phase-2 reasoner (we kill the server before eval though)

REASONER_PATH=/share/j_sun/lmlm_multihop/checkpoints/main/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak/checkpoint-500

KB_PKG=kb_baselines/autoschemakg
ATLAS_BASE_DIR="${KB_PKG}/output"
TAG="${DATASET}_${SPLIT}_n${N_QUESTIONS}_seed${SEED}"
INPUT_JSON="${ATLAS_BASE_DIR}/inputs/${TAG}.json"
ATLAS_OUTPUT_DIR="${ATLAS_BASE_DIR}/extractions/${TAG}"
KBEVO_KB_JSON="${ATLAS_BASE_DIR}/kbs/${TAG}_kb.json"
EVAL_OUTPUT_DIR="${ATLAS_BASE_DIR}/eval_runs"

mkdir -p "$(dirname "${INPUT_JSON}")" "${ATLAS_OUTPUT_DIR}" "$(dirname "${KBEVO_KB_JSON}")" "${EVAL_OUTPUT_DIR}"

echo "============================================================"
echo "[1/6] Build atlas-rag input JSON ($N_QUESTIONS questions)"
echo "============================================================"
PYTHONPATH=src python "${KB_PKG}/build_atlas_input.py" \
    --dataset "${DATASET}" --split "${SPLIT}" --setting "${SETTING}" \
    --n ${N_QUESTIONS} --seed ${SEED} \
    --use-contexts all \
    --output "${INPUT_JSON}"

echo "============================================================"
echo "[2/6] Launch vllm serve in background"
echo "============================================================"
VLLM_LOG="${ATLAS_OUTPUT_DIR}/vllm_serve.log"
nohup vllm serve "${EXTRACTOR_MODEL}" \
    --host ${VLLM_HOST} --port ${VLLM_PORT} \
    --gpu-memory-utilization ${VLLM_GPU_MEM} \
    --max-model-len 8192 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 16384 \
    --dtype bfloat16 \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
echo "vllm pid=${VLLM_PID}  log=${VLLM_LOG}"
trap "echo 'Cleaning up vllm pid=${VLLM_PID}'; kill ${VLLM_PID} 2>/dev/null || true" EXIT

# Wait for /v1/models to respond
echo "Waiting for vllm to come up..."
for i in $(seq 1 60); do
    if curl -s --max-time 2 "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        echo "vllm is ready after ${i}*10s"
        break
    fi
    sleep 10
    if [ $i -eq 60 ]; then
        echo "vllm failed to come up in 10 minutes. Last 50 lines of log:"
        tail -50 "${VLLM_LOG}"
        exit 1
    fi
done

echo "============================================================"
echo "[3/6] atlas-rag KG extraction (3 stages × 50 passages)"
echo "============================================================"
PYTHONPATH=src python "${KB_PKG}/run_atlas_extraction.py" \
    --input "${INPUT_JSON}" \
    --output-dir "${ATLAS_OUTPUT_DIR}" \
    --vllm-endpoint "http://${VLLM_HOST}:${VLLM_PORT}/v1" \
    --model-name "${EXTRACTOR_MODEL}" \
    --batch-size 64 \
    --max-workers 32

echo "============================================================"
echo "[4/6] Kill vllm server (free GPU for reasoner)"
echo "============================================================"
kill ${VLLM_PID} 2>/dev/null || true
wait ${VLLM_PID} 2>/dev/null || true
trap - EXIT
sleep 5

echo "============================================================"
echo "[5/6] Convert atlas-rag output -> KBevo unified KB JSON"
echo "============================================================"
PYTHONPATH=src python "${KB_PKG}/atlas_to_kbevo_kb.py" \
    --atlas-output-dir "${ATLAS_OUTPUT_DIR}" \
    --output "${KBEVO_KB_JSON}"

echo "============================================================"
echo "[6/6] Phase-2 eval with KBEVO-GRPO 1.7B + atlas-rag KB"
echo "============================================================"
PYTHONPATH=src python src/eval_multihop.py \
    --model-path "${REASONER_PATH}" \
    --method two_phase \
    --dataset "${DATASET}" --split "${SPLIT}" --setting "${SETTING}" \
    --total-count ${N_QUESTIONS} \
    --batch-size 64 \
    --seed ${SEED} \
    --save-every 64 \
    --start-index 0 \
    --top-k 4 \
    --similarity-threshold 0.6 \
    --use-train-params \
    --concat-all-db \
    --use-contexts all \
    --edc-kb-path "${KBEVO_KB_JSON}" \
    --output-dir "${EVAL_OUTPUT_DIR}" \
    --save-version "_smoke_atlas" \
    --eval \
    --resume

echo
echo "==== DONE ===="
echo "atlas-rag KB:    ${KBEVO_KB_JSON}"
echo "Phase-2 results: ${EVAL_OUTPUT_DIR}/two_phase/${DATASET}/"
echo "Look at:         find ${EVAL_OUTPUT_DIR} -name 'results_*.json'"
