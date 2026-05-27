#!/bin/bash
# EDC static-KG baseline (W2 rebuttal): per-dataset KB build (Qwen3-1.7B-Instruct
# vanilla, no SFT/GRPO) followed by Phase-2 eval with the KBEVO-GRPO (1.7B)
# checkpoint. Mirrors the main-table apples-to-apples setup:
#
#   --use-contexts all  (10 contexts split into 5 parts)
#   --concat-all-db     (cross-question unified KB, same as KBEVO main eval)
#   --use-train-params  T=1.0  top_p=0.95  vllm_top_k=4
#   --top-k 4           --similarity-threshold 0.6
#   NO --use-inverses   (KBEVO main eval's --use-inverses is a no-op due to a bug
#                        in build_databases_from_triplets_batch — we match the
#                        actual KBEVO behavior, not the nominal flag)
#
# Submits one sbatch per (dataset) pair, each running build_edc_kb.py once and
# then eval_multihop.py against that KB.
#
# Run on a login node:
#   bash menghan-scripts/eval_edc_baseline.slurm

cd /share/j_sun/mx253/Multi-Hop-Reasoning

# ── Configuration ────────────────────────────────────────────────────────────
NUM_SAMPLES=1000
SEED=42
KB_PKG=kb_baselines/edc
EDC_KB_DIR="${KB_PKG}/output/kbs"
OUTPUT_DIR="${KB_PKG}/output/eval_runs"
SAVE_VERSION="_edc0523"
EXTRACTOR="Qwen/Qwen3-1.7B"
PROMPT_KEY="edc_extract"

# Phase-2 reasoner = KBEVO-GRPO (1.7B), checkpoint-500
REASONER_PATH=/share/j_sun/lmlm_multihop/checkpoints/main/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak/checkpoint-500

# Retrieval / two_phase params (match training)
TOP_K=4
SIMILARITY_THRESHOLD=0.6

mkdir -p "${EDC_KB_DIR}"

DATASETS=(hotpotqa musique 2wiki)

for DATASET in "${DATASETS[@]}"; do
    EDC_KB_PATH="${EDC_KB_DIR}/edc_${DATASET}_dev_n${NUM_SAMPLES}_seed${SEED}.json"
    JOB_NAME="edc_${DATASET}"
    TMP_SCRIPT=$(mktemp /tmp/mx_edc_XXXXXX.sh)

    cat > "${TMP_SCRIPT}" << SCRIPT_EOF
#!/bin/bash
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate lmlm_b200
cd /share/j_sun/mx253/Multi-Hop-Reasoning

set -euo pipefail

# Step 1: build EDC KB (skip if file exists)
if [[ ! -f "${EDC_KB_PATH}" ]]; then
    echo "========== [1/2] Build EDC KB: ${DATASET} =========="
    PYTHONPATH=src python ${KB_PKG}/build_edc_kb.py \\
        --dataset "${DATASET}" \\
        --split dev \\
        --setting distractor \\
        --n ${NUM_SAMPLES} \\
        --seed ${SEED} \\
        --num-parts 5 \\
        --use-contexts all \\
        --extractor "${EXTRACTOR}" \\
        --prompt-key "${PROMPT_KEY}" \\
        --output "${EDC_KB_PATH}"
else
    echo "========== [1/2] EDC KB exists, skip: ${EDC_KB_PATH} =========="
fi

# Step 2: Phase-2 eval with KBEVO-GRPO (1.7B) reasoner against the EDC KB
echo "========== [2/2] Phase-2 eval with EDC KB: ${DATASET} =========="
PYTHONPATH=src python src/eval_multihop.py \\
    --model-path "${REASONER_PATH}" \\
    --method two_phase \\
    --dataset "${DATASET}" \\
    --split dev \\
    --setting distractor \\
    --total-count ${NUM_SAMPLES} \\
    --batch-size 64 \\
    --seed ${SEED} \\
    --save-every 64 \\
    --start-index 0 \\
    --top-k ${TOP_K} \\
    --similarity-threshold ${SIMILARITY_THRESHOLD} \\
    --use-train-params \\
    --concat-all-db \\
    --use-contexts all \\
    --edc-kb-path "${EDC_KB_PATH}" \\
    --output-dir "${OUTPUT_DIR}" \\
    --save-version "${SAVE_VERSION}" \\
    --eval \\
    --resume
SCRIPT_EOF

    echo "Submitting: ${JOB_NAME}"
    sbatch \
        -J "${JOB_NAME}" \
        -o "/home/mx253/slurm_output/lmlm_multihop/${JOB_NAME}_%j.out" \
        -e "/home/mx253/slurm_output/lmlm_multihop/${JOB_NAME}_%j.err" \
        -N 1 -n 1 \
        --cpus-per-task=8 \
        --mem=128G \
        -t 8:00:00 \
        --partition=aimi,default_partition \
        --exclude=jjs533-compute-01,jjs533-compute-02,jjs533-compute-03 \
        --gres=gpu:nvidia_rtx_a6000:1 \
        --requeue \
        "${TMP_SCRIPT}"
    rm -f "${TMP_SCRIPT}"
done

echo "All EDC baseline jobs submitted!"
