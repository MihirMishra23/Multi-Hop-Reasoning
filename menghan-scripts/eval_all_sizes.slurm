#!/bin/bash
# Launcher: eval 1.7B/4B/8B (SFT + GRPO) on hotpotqa/musique/2wiki.
# Adapted from scripts/lx_scripts/eval_lmlm.slurm.
# Submits one sbatch per (model, dataset) combo via per_ckpt_benchmark granularity.
#
# Run on a login node (not inside sbatch). This script itself is just a launcher;
# it will submit ~24 child jobs (8 models * 3 datasets).
#
# Usage:
#   bash menghan-scripts/eval_all_sizes.slurm

cd /share/j_sun/mx253/Multi-Hop-Reasoning

echo "Starting Eval launcher (mx253)..."

# ── Configuration ─────────────────────────────────────────────────────────────
EVAL_MODE="latest"             # "latest" | "all" | "target"
TARGET_CKPT=500
NUM_SAMPLES=1000
OUTPUT_DIR="./output/main_tables_mx"
SAVE_VERSION="_mx0520"

# Retrieval / two_phase params (match training)
TOP_K=4
SIMILARITY_THRESHOLD=0.6
USE_INVERSES="--use-inverses"
CONCAT_ALL_DB="--concat-all-db"
USE_CONTEXTS="all"
USE_TRAIN_PARAMS="--use-train-params"          # T=1.0  top_p=0.95  top_k=4
TWO_PHASE_METHOD="two_phase"

# ── Models (absolute paths — split between two roots) ────────────────────────
MODEL_PATH_LST=(
    # SFT
    /share/j_sun/lmlm_multihop/checkpoints/main/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.6_top_k_4
    /share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-4B-SFT_hotpotqa_ep3_bsz48_th-1
    /share/j_sun/lmlm_multihop/checkpoints/main/Qwen3-8B-SFT_hotpotqa_ep3_bsz48_th-1
    /share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-8B-SFT_hotpotqa_ep3_bsz48_lr5e-5_th-1
    # GRPO
    /share/j_sun/lmlm_multihop/checkpoints/main/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak
    /share/j_sun/lmlm_multihop/checkpoints/main/Qwen3-4B-SFT_hotpotqa_ep3_bsz48_th-1-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak
    /share/j_sun/lmlm_multihop/checkpoints/main/Qwen3-8B-SFT_hotpotqa_ep3_bsz48_th-1-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak
    /share/j_sun/lmlm_multihop/checkpoints/main/Qwen3-8B-SFT_hotpotqa_ep3_bsz48_lr5e-5_th-1-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak
)

DATASET_LST=(hotpotqa musique 2wiki)
SPLIT_LST=(dev)
SETTING="distractor"

# ── Compatibility check ───────────────────────────────────────────────────────
is_valid_combo() {
    local ds="$1" sp="$2"
    case "${ds}" in
        hotpotqa)       [[ "${sp}" =~ ^(dev|train_val1k)$ ]] ;;
        musique)        [[ "${sp}" == "dev" ]] ;;
        2wiki)          [[ "${sp}" == "dev" ]] ;;
        *)              return 1 ;;
    esac
}

# ── Main loop ─────────────────────────────────────────────────────────────────
for MODEL_DIR in "${MODEL_PATH_LST[@]}"; do
    # Resolve checkpoint dirs
    if [[ -d "${MODEL_DIR}" ]] && find "${MODEL_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -printf '.\n' 2>/dev/null | grep -q .; then
        # Has checkpoint-* subdirs
        mapfile -t CKPT_DIRS < <(
            find "${MODEL_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V
        )
        if [[ "${EVAL_MODE}" == "latest" ]]; then
            CKPT_DIRS=( "${CKPT_DIRS[-1]}" )
        elif [[ "${EVAL_MODE}" == "target" ]]; then
            CKPT_DIRS=( "checkpoint-${TARGET_CKPT}" )
        fi
    else
        # Model dir IS the checkpoint (SFT case with no checkpoint-* subdir)
        CKPT_DIRS=( "" )
    fi

    for CKPT in "${CKPT_DIRS[@]}"; do
        if [[ -n "${CKPT}" ]]; then
            MODEL_PATH="${MODEL_DIR}/${CKPT}"
            CKPT_SHORT="${CKPT#checkpoint-}"
        else
            MODEL_PATH="${MODEL_DIR}"
            CKPT_SHORT="sft"
        fi
        MODEL_NAME="$(basename "${MODEL_DIR}")"

        for DATASET in "${DATASET_LST[@]}"; do
            for SPLIT in "${SPLIT_LST[@]}"; do
                if ! is_valid_combo "${DATASET}" "${SPLIT}"; then
                    continue
                fi

                JOB_NAME="eval_${DATASET}_${MODEL_NAME:0:30}_ckpt${CKPT_SHORT}"
                TMP_SCRIPT=$(mktemp /tmp/mx_eval_XXXXXX.sh)

                cat > "${TMP_SCRIPT}" << SCRIPT_EOF
#!/bin/bash
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate lmlm_b200
cd /share/j_sun/mx253/Multi-Hop-Reasoning
echo "========== ${MODEL_PATH} | ${DATASET} | ${SPLIT} =========="
bash scripts/eval_lmlm_multihop.sh \
    --model_path "${MODEL_PATH}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --method "${TWO_PHASE_METHOD}" \
    --num_samples ${NUM_SAMPLES} \
    --save_version "${SAVE_VERSION}" \
    --output-dir "${OUTPUT_DIR}" \
    --top-k ${TOP_K} \
    --similarity-threshold ${SIMILARITY_THRESHOLD} \
    ${USE_INVERSES} \
    ${USE_TRAIN_PARAMS} \
    ${CONCAT_ALL_DB} \
    --use-contexts "${USE_CONTEXTS}" \
    --setting "${SETTING}"
SCRIPT_EOF

                echo "Submitting: ${JOB_NAME}"
                sbatch \
                    -J "${JOB_NAME}" \
                    -o "/home/mx253/slurm_output/lmlm_multihop/${JOB_NAME}_%j.out" \
                    -e "/home/mx253/slurm_output/lmlm_multihop/${JOB_NAME}_%j.err" \
                    -N 1 -n 1 \
                    --cpus-per-task=8 \
                    --mem=128G \
                    -t 6:00:00 \
                    --partition=aimi,default_partition \
                    --exclude=jjs533-compute-01,jjs533-compute-02,jjs533-compute-03 \
                    --gres=gpu:nvidia_rtx_a6000:1 \
                    --requeue \
                    "${TMP_SCRIPT}"
                rm -f "${TMP_SCRIPT}"
            done
        done
    done
done

echo "All eval jobs submitted!"