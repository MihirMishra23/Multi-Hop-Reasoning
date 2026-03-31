#!/bin/bash

# SLURM Directives
#SBATCH -J lmlm_eval
#SBATCH -o /home/lz586/icl/slurm_output/lmlm_multi_hop/lmlm_eval_%j.out
#SBATCH -e /home/lz586/icl/slurm_output/lmlm_multi_hop/lmlm_eval_%j.err
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

echo "Starting Eval..."

# ── Configuration ─────────────────────────────────────────────────────────────

EVAL_MODE="latest"   # "latest" = last checkpoint only, "all" = every checkpoint
SUBMIT_MODE="run"  # "launch" = one sbatch job per ckpt/dataset/split, "run" = sequential (legacy)

# Model roots
SFT_MODEL_ROOT=/share/j_sun/lmlm_multihop/checkpoints/main
SFT_MODEL_ROOT2=/share/j_sun/lz586/checkpoints/lmlm_multi_hop
GRPO_MODEL_ROOT=/share/j_sun/lmlm_multihop/checkpoints/debug/

# ── Models to evaluate (uncomment what you need) ─────────────────────────────
MODEL_PATH_LST=(
    # ----------------------------
    # Main CKPTs on hotpotqa
    # Qwen3-1.7B-SFT_ep5_bsz48
    # Qwen3-1.7B-SFT_ep5_bsz48-grpo-g8-bs16-s8-b0.0-ep3-n7000
    # Qwen3-1.7B-SFT_ep5_bsz48-grpo-g8-bs16-s8-b0.0-ep5-n8000
    # Qwen3-4B-SFT_ep5_bsz48
    # Qwen3-4B-SFT_ep5_bsz48-grpo-g8-bs16-s8-b0.0-ep5-n8000
    # ----------------------------

    # Ablation: F1 threshold
    # Qwen3-1.7B-SFT_ep5_bsz48_th0.8-grpo-g8-bs16-s8-b0.0-ep5-n8000
    # Qwen3-1.7B-SFT_ep5_bsz48_th-2-grpo-g8-bs16-s8-b0.0-ep5-n8000
    # Qwen3-1.7B-SFT_ep5_bsz48_th0.3-grpo-g8-bs16-s8-b0.0-ep5-n8000

    # Debug
    # Qwen3-1.7B-SFT_mquake-cf-debug_ep5_bsz48_th1.0
    # Qwen3-1.7B-SFT_mquake-real-debug_ep5_bsz48_th1.0

    # Ablation: base model version
    # Qwen2.5-3B-SFT_ep5_bsz48_th-1

    # Ablation: new db format, return triples
    # Qwen3-1.7B-SFT_ep5_bsz48_th-3

    # Ablation: LMLM-llama2-382M
    # LMLM-llama2-382M-SFT_ep5_bsz48_th-3
    # LMLM-llama2-382M-SFT_ep5_bsz48_th-1
    # LMLM-llama2-382M-SFT_ep5_bsz48_th-1-grpo-g8-bs256-s1-b0.0-ep5-n8000-nak
    # LMLM-llama2-382M-SFT_hotpotqa_ep10_bsz48_th-1

    # ----------------------------
    # Main CKPTs on mquake
    # Qwen3-1.7B-SFT_mquake_ep5_bsz48_th1.0
    # Qwen3-1.7B-SFT_mquake_ep5_bsz48_th1.0-grpo-g8-bs16-s8-b0.0-ep5-n5334-inv
    # ----------------------------

    # Ablation: training on mquake variants
    # Qwen3-1.7B-SFT_mquake-cf5334_ep5_bsz48_th1.0
    # Qwen3-1.7B-SFT_mquake-real_ep5_bsz48_th1.0
    # Qwen3-1.7B-SFT_mquake-cf_ep5_bsz48_th1.0

    # Ablation: augmentation variants
    # Qwen3-1.7B-SFT_mquake-cf5334-r10_ep5_bsz48_th1.0
    # Qwen3-1.7B-SFT_mquake-cf5334-r1_ep5_bsz48_th1.0
    # Qwen3-1.7B-SFT_mquake-cfall-r10_ep5_bsz48_th1.0
    # Qwen3-1.7B-SFT_mquake-real5334-r1_ep5_bsz48_th1.0
    # ----------------------------

    # Two-phase SFT
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k
    # Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed

    # Two-phase GRPO - full runs
    # Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed-grpo-B1-K4-M4-bs16-s8-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak

    # Two-phase GRPO - n1000
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B1-K4-M4-bs16-s16-b0.0-ep5-n1000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B2-K4-M2-bs16-s8-b0.0-ep5-n1000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B0-K4-M8-bs16-s8-b0.0-ep5-n1000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B1-K8-M2-bs16-s8-b0.0-ep5-n1000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B8-K4-M8-bs16-s16-b0.0-ep5-n1000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak

    # Two-phase GRPO - n500
    # Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed-grpo-B2-K4-M2-bs16-s4-b0.0-ep10-n500-f1-v2-p1binary-ptcontext_only-wmcount_dynamic-th0.6-topk4-debug-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed-grpo-B2-K4-M2-bs16-s4-b0.0-ep10-n500-em_size-v2-p1binary-ptcontext_only-wmcount_dynamic-th0.6-topk4-debug-nak

    # Two-phase GRPO - new full runs
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N8-K4-B64-M2-b0.0-step500-n7000-f1-2ph-prcontext_only-wcount_dynamic-th0.6-topk4-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-step500-n7000-f1-2ph-prcontext_only-wcount-th0.6-topk4-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-step500-n7000-em-2ph-prcontext_only-wcount-th0.6-topk4-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs1024-N32-K4-B32-M7-b0.0-step500-n7000-em-2ph-prcontext_only-wcount-th0.6-topk4-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-step500-n7000-f1-v2-2ph-prcontext_only-wcount-th0.6-topk4-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B32-K2-M4-bs16-s16-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak

    # Menghan sweeps
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B32-K4-M8-bs16-s64-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B16-K4-M8-bs16-s32-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B64-K2-M4-bs16-s32-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B128-K2-M4-bs16-s64-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B16-K8-M4-bs16-s32-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B8-K4-M8-bs16-s16-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B32-K4-M4-bs16-s32-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B32-K2-M4-bs16-s16-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak


    # Currently active
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.6_top_k_4
    # Qwen3-4B-SFT_hotpotqa_ep3_bsz48_th-1

    # Main 2-phase full runs
    Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak
    # Qwen3-4B-SFT_hotpotqa_ep3_bsz48_th-1-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak

    # Ablation: em_size, em_size_4b,lr=1e-6
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-em_size-2ph-prsft-wcount-th0.6-topk4-nak
    # Qwen3-4B-SFT_hotpotqa_ep3_bsz48_th-1-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-em_size-2ph-prsft-wcount-th0.6-topk4-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-step500-n7000-f1-2ph-prcontext_only-wcount-th0.6-topk4-nak

    # # Ablation: early stopping, zero rl, 1-phase only, vanilla grpo
    # phase 1 final
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_1phase-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-th0.6-topk4-nak
    # # early sft
    # checkpoint-92-grpo-tbs512-N16-K4-B32-M3-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak
    # checkpoint-276-grpo-tbs512-N16-K4-B32-M3-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak
    # # N16
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.6_top_k_4-grpo-tbs512-N16-K4-B32-M3-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak

    # # top1-th0.9
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.9_top_k_1-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.9-topk1-nak

    # 
    # checkpoint-368-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-th0.6-topk4-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-vanilla-th0.6-topk4-nak

    # Ablation: topk=1 (todo)


    # Zero RL
    # Qwen3-4B-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1_format-2ph-prformatted_zero_rl_v7-wcount-th0.6-topk4-nak/checkpoint-100
    # Qwen3-1.7B-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1_format-2ph-prformatted_zero_rl_v5-wcount-th0.6-topk4-nak/checkpoint-100
    # Qwen3-1.7B-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prformatted_zero_rl_v5-wcount-th0.6-topk4-nak
)

# TWO_PHASE_METHOD="lmlm"
TWO_PHASE_METHOD="two_phase"

# ── Datasets and splits ──────────────────────────────────────────────────────
# NOTE: train_val1k is only valid for hotpotqa. musique/2wiki only support "dev".
#       mquake supports: eval-original, eval-edit, eval-edit-new, train
DATASET_LST=(
    # synthworlds
    # trivia_qa
    # hotpotqa
    # musique
    # 2wiki
    mquake
)
SPLIT_LST=(
    # train_val1k
    # dev
    # eval-original
    eval-edit
)
# if synthworlds, qa-sm, qa-rm, otherwise distractor
if [[ "${DATASET_LST[@]}" == *"synthworlds"* ]]; then
    SETTING_LST=(
        qa-sm
        # qa-rm
    )
else
    SETTING_LST=(
        distractor
    )
fi


# ── Eval parameters ──────────────────────────────────────────────────────────
OUTPUT_DIR="./output/main_tables"
SAVE_VERSION="_0330"
NUM_SAMPLES=1000
TOP_K_LST=(4)
SIMILARITY_THRESHOLD=0.6

# Sampling: leave empty for greedy (temperature=0, recommended for eval).
# Set to "--use-train-params" to use training sampling (T=1.0, top_p=0.95).
# USE_TRAIN_PARAMS="--use-train-params"
USE_TRAIN_PARAMS_LST=("--use-train-params")
# USE_TRAIN_PARAMS_LST=("")

# Two-phase only
CONCAT_ALL_DB_LST=("--concat-all-db")
USE_CONTEXTS_LST=("all")              # "golden" or "all"
# USE_CONTEXTS_LST=("golden" "all")              # "golden" or "all"

# Retrieval flags
ADAPTIVE_K=""                   # "--adaptive-k" to enable
USE_INVERSES="--use-inverses"

# ── Compatibility check ──────────────────────────────────────────────────────
is_valid_combo() {
    local ds="$1" sp="$2"
    case "${ds}" in
        hotpotqa)           [[ "${sp}" =~ ^(dev|train_val1k|train_train1k)$ ]] ;;
        musique)            [[ "${sp}" == "dev" ]] ;;
        2wiki|two_wiki)     [[ "${sp}" == "dev" ]] ;;
        mquake*)            [[ "${sp}" =~ ^(eval-edit|eval-edit-new|eval-original|train)$ ]] ;;
        trivia_qa)          [[ "${sp}" == "dev" ]] ;;
        synthworlds)        [[ "${sp}" == "dev" ]] ;;
        *)                  return 1 ;;
    esac
}

# ── Main loop ─────────────────────────────────────────────────────────────────
for SETTING in "${SETTING_LST[@]}"; do
for TOP_K in "${TOP_K_LST[@]}"; do
for USE_TRAIN_PARAMS in "${USE_TRAIN_PARAMS_LST[@]}"; do
for CONCAT_ALL_DB in "${CONCAT_ALL_DB_LST[@]}"; do
for USE_CONTEXTS in "${USE_CONTEXTS_LST[@]}"; do
for MODEL_DIR in "${MODEL_PATH_LST[@]}"; do

    # Resolve model root
    if [[ "${MODEL_DIR}" == *"grpo"* ]]; then
        MODEL_ROOT="${GRPO_MODEL_ROOT}"
    elif [[ "${MODEL_DIR}" == *"1.7B"* ]]; then
        MODEL_ROOT="${SFT_MODEL_ROOT}"
    else
        MODEL_ROOT="${SFT_MODEL_ROOT2}"
    fi

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
    fi

    for CKPT in "${CKPT_DIRS[@]}"; do
        MODEL_PATH="${MODEL_ROOT}/${MODEL_DIR}/${CKPT}"

        # Write a temp script for this checkpoint (avoids all quoting issues with --wrap)
        TMP_SCRIPT=$(mktemp /tmp/lmlm_eval_XXXXXX.sh)
        cat > "${TMP_SCRIPT}" << SCRIPT_EOF
#!/bin/bash
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mem
cd /home/lz586/icl/Multi-Hop-Reasoning/
SCRIPT_EOF

        for SPLIT in "${SPLIT_LST[@]}"; do
            for DATASET in "${DATASET_LST[@]}"; do
                if ! is_valid_combo "${DATASET}" "${SPLIT}"; then
                    continue
                fi
                if [[ "${DATASET}" == "mquake"* ]]; then
                    METHOD="lmlm"
                else
                    METHOD="${TWO_PHASE_METHOD}"
                fi

                    # Automatically set setting to 'qa-sm' for synthworlds dataset
                if [[ "${DATASET}" == "synthworlds" ]]; then
                    USE_CONTEXTS="golden"
                    echo "NOTE: Automatically setting --use-contexts=golden for synthworlds dataset (only golden contexts are available)"
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
    --use-contexts "${USE_CONTEXTS}" \
    --setting "${SETTING}"
SCRIPT_EOF
            done
        done

        if [[ "${SUBMIT_MODE}" == "launch" ]]; then
            JOB_NAME="lmlm_${MODEL_DIR}_${TOP_K}_${USE_TRAIN_PARAMS}_${CONCAT_ALL_DB}_${USE_CONTEXTS}"
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
                --partition=kilian \
                --gres=gpu:nvidia_rtx_a6000:1 \
                --requeue \
                "${TMP_SCRIPT}"
        else
            bash "${TMP_SCRIPT}"
            rm -f "${TMP_SCRIPT}"
        fi
    done
done
done
done
done
done
done

echo "Evaluation complete!"