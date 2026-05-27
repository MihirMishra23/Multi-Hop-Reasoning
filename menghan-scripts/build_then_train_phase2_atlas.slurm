#!/bin/bash
# Experiment B end-to-end: build atlas-rag train KB → train Phase-2-only GRPO.
#
# Runs SEQUENTIALLY in one sbatch (same B200 GPU allocation for both stages).
# Total budget: 24h (build ~3-4h + train ~10-12h + overhead).
#
# Stage A (build_atlas_kb_train.slurm body):
#   1) build input JSON (CPU — actually pre-built; just re-runs harmlessly)
#   2) start vllm serve Llama-3.1-8B-Instruct
#   3) atlas-rag KG extraction (3 stages)
#   4) kill vllm
#   5) convert atlas-rag output → KBevo unified KB JSON
#
# Stage B (train_phase2_only_atlas.slurm body):
#   - bash scripts/grpo_train.sh ... (Phase-2-only with atlas-rag KB)
#
# Safety: Stage B checks for KB file presence first; if Stage A failed and KB
# is missing, Stage B exits with error (no wasted GRPO compute on stale data).
#
# Usage:
#   sbatch menghan-scripts/build_then_train_phase2_atlas.slurm

#SBATCH --job-name=expB-buildtrain-atlas-1.7b
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/expB_buildtrain_atlas_%j.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/expB_buildtrain_atlas_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --partition=aimi,default_partition
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --exclude=jjs533-compute-01,jjs533-compute-02,jjs533-compute-03
#SBATCH --requeue

set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate lmlm_b200
cd /share/j_sun/mx253/Multi-Hop-Reasoning

# ═════════════════════════════════════════════════════════════════════════════
# STAGE A: build atlas-rag train KB (~3-4h)
# ═════════════════════════════════════════════════════════════════════════════

# --- Slice (matches Gemini Phase-2-only training KB) ---
DATASET=hotpotqa
SPLIT=train
SETTING=distractor
SEED=42
START_IDX=82347
N=8100

EXTRACTOR_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
VLLM_PORT=8765
VLLM_HOST="127.0.0.1"
VLLM_GPU_MEM=0.55
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

mkdir -p "$(dirname ${INPUT_JSON})" "${ATLAS_OUTPUT_DIR}" "$(dirname ${KBEVO_KB_JSON})"

if [[ -f "${KBEVO_KB_JSON}" ]]; then
    echo "════════════════════════════════════════════════════════════════════"
    echo "[STAGE A] KB already exists at ${KBEVO_KB_JSON} — SKIPPING build."
    echo "════════════════════════════════════════════════════════════════════"
else
    echo "════════════════════════════════════════════════════════════════════"
    echo "[STAGE A 1/5] Build atlas-rag input JSON (${DATASET} TRAIN [${START_IDX}, ${START_IDX}+${N}))"
    echo "════════════════════════════════════════════════════════════════════"
    PYTHONPATH=src python ${KB_PKG}/build_atlas_input.py \
        --dataset "${DATASET}" --split "${SPLIT}" --setting "${SETTING}" \
        --n ${N} --seed ${SEED} --start-index ${START_IDX} \
        --use-contexts all \
        --output "${INPUT_JSON}"

    echo "════════════════════════════════════════════════════════════════════"
    echo "[STAGE A 2/5] Launch vllm serve (Llama-3.1-8B-Instruct, background)"
    echo "════════════════════════════════════════════════════════════════════"
    VLLM_LOG="${ATLAS_OUTPUT_DIR}/vllm_serve.log"
    nohup vllm serve "${EXTRACTOR_MODEL}" \
        --host ${VLLM_HOST} --port ${VLLM_PORT} \
        --gpu-memory-utilization ${VLLM_GPU_MEM} \
        --max-model-len 8192 \
        --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
        --max-num-batched-tokens ${VLLM_MAX_BATCHED_TOKENS} \
        --dtype bfloat16 \
        > "${VLLM_LOG}" 2>&1 &
    VLLM_PID=$!
    echo "vllm pid=${VLLM_PID}  log=${VLLM_LOG}"
    trap "echo 'Cleaning up vllm pid=${VLLM_PID}'; kill ${VLLM_PID} 2>/dev/null || true" EXIT

    echo "Waiting for vllm to come up..."
    for i in $(seq 1 60); do
        if curl -s --max-time 2 "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
            echo "vllm is ready after ${i}*10s"; break
        fi
        sleep 10
        if [ $i -eq 60 ]; then
            echo "vllm failed to come up in 10 minutes."
            tail -50 "${VLLM_LOG}"; exit 1
        fi
    done

    echo "════════════════════════════════════════════════════════════════════"
    echo "[STAGE A 3/5] atlas-rag KG extraction (3 stages, ~80k passages)"
    echo "════════════════════════════════════════════════════════════════════"
    PYTHONPATH=src python ${KB_PKG}/run_atlas_extraction.py \
        --input "${INPUT_JSON}" \
        --output-dir "${ATLAS_OUTPUT_DIR}" \
        --vllm-endpoint "http://${VLLM_HOST}:${VLLM_PORT}/v1" \
        --model-name "${EXTRACTOR_MODEL}" \
        --batch-size ${ATLAS_BATCH_SIZE} \
        --max-workers ${ATLAS_MAX_WORKERS}

    echo "════════════════════════════════════════════════════════════════════"
    echo "[STAGE A 4/5] Kill vllm server (free GPU for trainer)"
    echo "════════════════════════════════════════════════════════════════════"
    kill ${VLLM_PID} 2>/dev/null || true
    wait ${VLLM_PID} 2>/dev/null || true
    trap - EXIT
    sleep 15  # extra buffer to ensure GPU memory fully freed

    echo "════════════════════════════════════════════════════════════════════"
    echo "[STAGE A 5/5] Convert atlas-rag output -> KBevo unified KB JSON"
    echo "════════════════════════════════════════════════════════════════════"
    PYTHONPATH=src python ${KB_PKG}/atlas_to_kbevo_kb.py \
        --atlas-output-dir "${ATLAS_OUTPUT_DIR}" \
        --output "${KBEVO_KB_JSON}"

    echo
    echo "==== STAGE A DONE: ${KBEVO_KB_JSON} ===="
fi

# ═════════════════════════════════════════════════════════════════════════════
# STAGE B: train Phase-2-only GRPO with atlas-rag KB (~10-12h)
# ═════════════════════════════════════════════════════════════════════════════

# Safety: refuse to start training if KB missing
if [[ ! -f "${KBEVO_KB_JSON}" ]]; then
    echo "ERROR: atlas-rag train KB not found at ${KBEVO_KB_JSON}" >&2
    echo "Stage A must have failed. Refusing to start training."
    exit 1
fi

# --- Training config (mirrors scripts/lx_scripts/grpo_training_1phase.slurm) ---
TRAIN_MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_1phase
TRAIN_DATASET_NAME=hotpotqa
TRAIN_SIZE=7000
TOTAL_BATCH_SIZE=512
TRAIN_RETRIEVAL_THRESHOLD=0.6
TRAIN_TOP_K=4
REWARD_FUNC="f1"
PHASE1_PROMPT_TYPE="sft"
# Save under atlas-specific dir to AVOID overwriting Linxi's Gemini 1phase ckpt
TRAIN_SAVE_DIR=/share/j_sun/lmlm_multihop/checkpoints/atlas_rag

if [[ ! -d "${TRAIN_MODEL_PATH}" ]]; then
    echo "ERROR: 1phase SFT base not found at ${TRAIN_MODEL_PATH}" >&2
    exit 1
fi

mkdir -p "${TRAIN_SAVE_DIR}"

echo "════════════════════════════════════════════════════════════════════"
echo "[STAGE B] GRPO Phase-2-only training: atlas-rag KB"
echo "  SFT base:  ${TRAIN_MODEL_PATH##*/}"
echo "  KB:        ${KBEVO_KB_JSON}"
echo "  Save dir:  ${TRAIN_SAVE_DIR}"
echo "  Slice:     hotpotqa train, seed=42, [82347, 90447)"
echo "  --two_phase: OFF (single-phase mode)"
echo "════════════════════════════════════════════════════════════════════"

bash scripts/grpo_train.sh \
    --gpu_type B200 \
    --model_path "${TRAIN_MODEL_PATH}" \
    --dataset_name "${TRAIN_DATASET_NAME}" \
    --database_path "${KBEVO_KB_JSON}" \
    --save_dir "${TRAIN_SAVE_DIR}" \
    --train_size ${TRAIN_SIZE} \
    --reward_func ${REWARD_FUNC} \
    --total_batch_size ${TOTAL_BATCH_SIZE} \
    --phase1_prompt_type ${PHASE1_PROMPT_TYPE} \
    --retrieval_threshold ${TRAIN_RETRIEVAL_THRESHOLD} \
    --top_k ${TRAIN_TOP_K} \
    --save_steps 50

echo
echo "════════════════════════════════════════════════════════════════════"
echo "==== EXPERIMENT B DONE ===="
echo "════════════════════════════════════════════════════════════════════"
echo "Final ckpt under: ${TRAIN_SAVE_DIR}/Qwen3-1.7B-SFT_..._1phase-grpo-...-nak/"
echo "Next: write eval slurm pointing at that ckpt's checkpoint-500"
