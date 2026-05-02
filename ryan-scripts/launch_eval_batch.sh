#!/bin/bash

# Launch multiple evaluation jobs with different configurations

echo "Submitting evaluation jobs..."
echo ""

# Configuration arrays - modify these as needed
DATASETS=(
    # "popqa"
    # "trivia_qa"
    # "hotpotqa"
    # "2wiki"
    # "musique"
    confiqa
)

NUM_SAMPLES_LST=(
    1000
)

SEEDS=(
    42
)

MODELS=(
    #SFT
    # "/share/j_sun/lmlm_multihop/checkpoints/main/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.9_top_k_1/checkpoint-735/"

    "/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-4B-SFT_hotpotqa_ep3_bsz48_th-1/checkpoint-735/"

    #GRPO
    # "/share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak/checkpoint-500/"
    # "/share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-4B-SFT_hotpotqa_ep3_bsz48_th-1-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak/checkpoint-500/"

)

# Optional: Override default parameters by uncommenting and modifying
OUTPUT_DIR="./643am-cf-100-rerun"
# SPLIT="train_val1k"
# SETTING="qa-sm"
CONFIQA_SETTING="cf_100"

# Build export string for optional parameters
EXTRA_EXPORT="ALL"
if [[ -n "${OUTPUT_DIR:-}" ]]; then
    EXTRA_EXPORT="${EXTRA_EXPORT},OUTPUT_DIR=${OUTPUT_DIR}"
fi
if [[ -n "${SPLIT:-}" ]]; then
    EXTRA_EXPORT="${EXTRA_EXPORT},SPLIT=${SPLIT}"
fi
if [[ -n "${SETTING:-}" ]]; then
    EXTRA_EXPORT="${EXTRA_EXPORT},SETTING=${SETTING}"
fi
if [[ -n "${CONFIQA_SETTING:-}" ]]; then
    EXTRA_EXPORT="${EXTRA_EXPORT},CONFIQA_SETTING=${CONFIQA_SETTING}"
fi

# Counter for job submission
JOB_COUNT=0

# Submit jobs for each combination
for DATASET in "${DATASETS[@]}"; do
    for NUM_SAMPLES in "${NUM_SAMPLES_LST[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            for MODEL in "${MODELS[@]}"; do
                # Extract the model directory name (parent of checkpoint-*)
                MODEL_DIR=$(basename "$(dirname "${MODEL}")")
                CHECKPOINT_DIR=$(basename "${MODEL}")
                MODEL_NAME="${MODEL_DIR}_${CHECKPOINT_DIR}"
                JOB_NAME="eval-${DATASET}-n${NUM_SAMPLES}-s${SEED}-${MODEL_NAME}"

                sbatch --job-name="${JOB_NAME}" \
                       --export="${EXTRA_EXPORT},DATASET=${DATASET},NUM_SAMPLES=${NUM_SAMPLES},SEED=${SEED},MODEL_PATH=${MODEL}" \
                       ryan-scripts/eval_batch.slurm

                ((JOB_COUNT++))
                echo "✓ Submitted: ${JOB_NAME}"
                echo "  Dataset: ${DATASET}, Samples: ${NUM_SAMPLES}, Seed: ${SEED}"
                echo "  Model: ${MODEL}"
                echo ""
            done
        done
    done
done

echo "=========================================="
echo "Total jobs submitted: ${JOB_COUNT}"
echo "Check status with: squeue -u $USER"
echo "=========================================="
