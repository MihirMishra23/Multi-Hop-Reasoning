#!/bin/bash

# SLURM Directives
#SBATCH -J lmlm_eval_qwen25
#SBATCH -o /home/lz586/icl/slurm_output/lmlm_multi_hop/lmlm_eval_qwen25_%j.out
#SBATCH -e /home/lz586/icl/slurm_output/lmlm_multi_hop/lmlm_eval_qwen25_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --get-user-env
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t infinite
#SBATCH --partition=aimi,kilian,jjs533,default_partition
#SBATCH --gres=gpu:nvidia_rtx_a6000:1

source activate mem
cd /home/lz586/icl/Multi-Hop-Reasoning/

echo "Starting Eval (Qwen2.5 debug checkpoints)..."

# ── Configuration ─────────────────────────────────────────────────────────────

EVAL_MODE="latest"   # "latest" = last checkpoint only, "all" = every checkpoint, "target" = specific checkpoint
TARGET_CKPT=500      # used when EVAL_MODE="target"
SUBMIT_MODE="launch"  # "launch" = submit via sbatch, "run" = sequential (legacy)
SUBMIT_GRANULARITY="per_ckpt_benchmark"  # "per_ckpt" | "per_benchmark" | "per_ckpt_benchmark"

# Model roots — Qwen2.5 final GRPO runs live under debug/, not main/
SFT_MODEL_ROOT=/share/j_sun/lmlm_multihop/checkpoints/debug
GRPO_MODEL_ROOT=/share/j_sun/lmlm_multihop/checkpoints/debug

# ── Models to evaluate ───────────────────────────────────────────────────────
MODEL_PATH_LST=(
    # Qwen2.5 final GRPO (identical hparams to Qwen3 final runs: tbs512-N32-K4-B16-M7-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak)
    Qwen2.5-3B-SFT_hotpotqa_ep3_bsz48_lr5e-5_th-1-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak
    Qwen2.5-1.5B-SFT_hotpotqa_ep3_bsz48_lr5e-5_th-1-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak
)

# TWO_PHASE_METHOD="lmlm"
TWO_PHASE_METHOD="two_phase"

# ── Datasets and splits ──────────────────────────────────────────────────────
# Match the Qwen3 final-runs eval suite: hotpotqa (train_val1k), 2wiki (dev), musique (dev).
DATASET_LST=(
    hotpotqa
    musique
    2wiki
    popqa
)
SPLIT_LST=(
    train_val1k
    dev
)
# if synthworlds: qa-sm / qa-rm; otherwise: distractor
if [[ "${DATASET_LST[@]}" == *"synthworlds"* ]]; then
    SETTING_LST=(qa-sm qa-rm)
else
    SETTING_LST=(distractor)
fi

# ── Eval parameters ──────────────────────────────────────────────────────────
OUTPUT_DIR="./output/main_tables"
SAVE_VERSION="_0330"
NUM_SAMPLES=1000

# Retrieval hyper-parameters (match training)
TOP_K=4
SIMILARITY_THRESHOLD=0.6
ADAPTIVE_K=""                    # "--adaptive-k" to enable
USE_INVERSES="--use-inverses"

# DB settings (two-phase only)
CONCAT_ALL_DB="--concat-all-db"  # "--concat-all-db" or ""
USE_CONTEXTS="all"               # "golden" or "all"

# Sampling: "" = greedy (temperature=0, recommended); "--use-train-params" = T=1.0, top_p=0.95
USE_TRAIN_PARAMS="--use-train-params"

# ── Compatibility check ──────────────────────────────────────────────────────
is_valid_combo() {
    local ds="$1" sp="$2"
    case "${ds}" in
        hotpotqa)           [[ "${sp}" =~ ^(dev|train_val1k|train_train1k)$ ]] ;;
        musique)            [[ "${sp}" == "dev" ]] ;;
        2wiki|two_wiki)     [[ "${sp}" == "dev" ]] ;;
        mquake*)            [[ "${sp}" =~ ^(eval-edit|eval-edit-new|eval-original|train)$ ]] ;;
        trivia_qa)          [[ "${sp}" == "dev" ]] ;;
        popqa)              [[ "${sp}" == "dev" ]] ;;
        synthworlds)        [[ "${sp}" == "dev" ]] ;;
        *)                  return 1 ;;
    esac
}

# ── Main loop ─────────────────────────────────────────────────────────────────
# Keys map group → temp script path / job name, enabling flexible granularity.
declare -A GROUPED_SCRIPTS
declare -A GROUPED_NAMES

for MODEL_DIR in "${MODEL_PATH_LST[@]}"; do

    # Resolve model root
    if [[ "${MODEL_DIR}" == *"grpo"* ]]; then
        MODEL_ROOT="${GRPO_MODEL_ROOT}"
    else
        MODEL_ROOT="${SFT_MODEL_ROOT}"
    fi

    # If MODEL_DIR already ends with /checkpoint-N, split it out so naming is consistent.
    # EVAL_MODE is intentionally ignored for explicit checkpoints.
    if [[ "${MODEL_DIR}" == */checkpoint-* ]]; then
        _CKPT_PART="${MODEL_DIR##*/}"          # e.g. checkpoint-500
        MODEL_DIR="${MODEL_DIR%/*}"            # strip trailing /checkpoint-N
        # Re-resolve MODEL_ROOT now that MODEL_DIR no longer contains the checkpoint suffix
        if [[ "${MODEL_DIR}" == *"grpo"* ]]; then
            MODEL_ROOT="${GRPO_MODEL_ROOT}"
        else
            MODEL_ROOT="${SFT_MODEL_ROOT}"
        fi
        if [[ ! -d "${MODEL_ROOT}/${MODEL_DIR}/${_CKPT_PART}" ]]; then
            echo "WARN: ${MODEL_ROOT}/${MODEL_DIR}/${_CKPT_PART} not found" >&2
            continue
        fi
        CKPT_DIRS=( "${_CKPT_PART}" )
    else
        # Find checkpoint dirs
        mapfile -t CKPT_DIRS < <(
            find "${MODEL_ROOT}/${MODEL_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
            | sort -V
        )
        if [[ ${#CKPT_DIRS[@]} -eq 0 ]]; then
            echo "WARN: No checkpoint-* dirs under ${MODEL_ROOT}/${MODEL_DIR}" >&2
            continue
        fi
        if [[ "${EVAL_MODE}" == "latest" ]]; then
            CKPT_DIRS=( "${CKPT_DIRS[-1]}" )
        elif [[ "${EVAL_MODE}" == "target" ]]; then
            CKPT_DIRS=( "checkpoint-${TARGET_CKPT}" )
            if [[ ! -d "${MODEL_ROOT}/${MODEL_DIR}/checkpoint-${TARGET_CKPT}" ]]; then
                echo "WARN: checkpoint-${TARGET_CKPT} not found under ${MODEL_ROOT}/${MODEL_DIR}" >&2
                continue
            fi
        fi
    fi

    for CKPT in "${CKPT_DIRS[@]}"; do
        MODEL_PATH="${MODEL_ROOT}/${MODEL_DIR}/${CKPT}"
        CKPT_SHORT="${CKPT#checkpoint-}"
        MODEL_DIR_SAFE="${MODEL_DIR//\//_}"

        for DATASET in "${DATASET_LST[@]}"; do

            # Determine group key and job name based on SUBMIT_GRANULARITY
            case "${SUBMIT_GRANULARITY}" in
                per_ckpt)
                    GROUP_KEY="${MODEL_PATH}"
                    GROUP_NAME="lmlm_eval_${MODEL_DIR_SAFE}_ckpt${CKPT_SHORT}"
                    ;;
                per_benchmark)
                    GROUP_KEY="${DATASET}"
                    GROUP_NAME="lmlm_eval_${DATASET}"
                    ;;
                *)  # per_ckpt_benchmark
                    GROUP_KEY="${MODEL_PATH}__${DATASET}"
                    GROUP_NAME="lmlm_eval_${DATASET}_${MODEL_DIR_SAFE}_ckpt${CKPT_SHORT}"
                    ;;
            esac

            # Initialize temp script for this group if not yet created
            if [[ -z "${GROUPED_SCRIPTS[${GROUP_KEY}]+x}" ]]; then
                TMP_SCRIPT=$(mktemp /tmp/lmlm_eval_XXXXXX.sh)
                GROUPED_SCRIPTS["${GROUP_KEY}"]="${TMP_SCRIPT}"
                GROUPED_NAMES["${GROUP_KEY}"]="${GROUP_NAME}"
                cat > "${TMP_SCRIPT}" << SCRIPT_EOF
#!/bin/bash
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mem
cd /home/lz586/icl/Multi-Hop-Reasoning/
SCRIPT_EOF
            fi
            TMP_SCRIPT="${GROUPED_SCRIPTS[${GROUP_KEY}]}"

            for SETTING in "${SETTING_LST[@]}"; do
                for SPLIT in "${SPLIT_LST[@]}"; do
                    if ! is_valid_combo "${DATASET}" "${SPLIT}"; then
                        continue
                    fi
                    if [[ "${DATASET}" == "mquake"* ]]; then
                        METHOD="lmlm"
                    else
                        METHOD="${TWO_PHASE_METHOD}"
                    fi

                    # Synthworlds always uses golden context
                    _USE_CONTEXTS="${USE_CONTEXTS}"
                    if [[ "${DATASET}" == "synthworlds" ]]; then
                        _USE_CONTEXTS="golden"
                    fi

                    cat >> "${TMP_SCRIPT}" << SCRIPT_EOF
echo "========== ${MODEL_PATH} | ${DATASET} | ${SPLIT} | ${METHOD} =========="
bash /home/lz586/icl/Multi-Hop-Reasoning/scripts/eval_lmlm_multihop.sh \
    --model_path "${MODEL_PATH}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --method "${METHOD}" \
    --num_samples ${NUM_SAMPLES} \
    --save_version "${SAVE_VERSION}" \
    --output-dir "${OUTPUT_DIR}" \
    --top-k ${TOP_K} \
    --similarity-threshold ${SIMILARITY_THRESHOLD} \
    ${ADAPTIVE_K} \
    ${USE_INVERSES} \
    ${USE_TRAIN_PARAMS} \
    ${CONCAT_ALL_DB} \
    --use-contexts "${_USE_CONTEXTS}" \
    --setting "${SETTING}"
SCRIPT_EOF
                done  # SPLIT
            done  # SETTING

        done  # DATASET
    done  # CKPT
done  # MODEL_DIR

# ── Submit / run all grouped scripts ─────────────────────────────────────────
for GROUP_KEY in "${!GROUPED_SCRIPTS[@]}"; do
    TMP_SCRIPT="${GROUPED_SCRIPTS[${GROUP_KEY}]}"
    JOB_NAME="${GROUPED_NAMES[${GROUP_KEY}]}"
    if [[ "${SUBMIT_MODE}" == "launch" ]]; then
        echo "Submitting: ${JOB_NAME}"
        sbatch \
            -J "${JOB_NAME}" \
            -o "/home/lz586/icl/slurm_output/lmlm_multi_hop/${JOB_NAME}_%j.out" \
            -e "/home/lz586/icl/slurm_output/lmlm_multi_hop/${JOB_NAME}_%j.err" \
            -N 1 -n 1 \
            --get-user-env \
            --cpus-per-task=4 \
            --mem=64G \
            -t infinite \
            --partition=jjs533 \
            --gres=gpu:nvidia_rtx_pro_6000_blackwell_max-q_workstation_edition:1 \
            --requeue \
            "${TMP_SCRIPT}"
        rm -f "${TMP_SCRIPT}"
    else
        bash "${TMP_SCRIPT}"
        rm -f "${TMP_SCRIPT}"
    fi
done

echo "Evaluation complete!"
