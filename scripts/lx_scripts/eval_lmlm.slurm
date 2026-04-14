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

EVAL_MODE="latest"   # "latest" = last checkpoint only, "all" = every checkpoint, "target" = specific checkpoint
TARGET_CKPT=400      # used when EVAL_MODE="target"
SUBMIT_MODE="launch"  # "launch" = submit via sbatch, "run" = sequential (legacy)
SUBMIT_GRANULARITY="per_ckpt_benchmark"  # "per_ckpt" | "per_benchmark" | "per_ckpt_benchmark"

# Model roots
SFT_MODEL_ROOT=/share/j_sun/lmlm_multihop/checkpoints/main
GRPO_MODEL_ROOT=/share/j_sun/lmlm_multihop/checkpoints/main

# ── Models to evaluate (uncomment what you need) ─────────────────────────────
MODEL_PATH_LST=(
    ## -- Version 1: 1-phase full runs Jan 2026 --
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

    ## -- Version 2: 2-phase full runs Feb 2026 --
    # ----------------------------
    ### Test Runs
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

    # ----------------------------
    # Final Runs: SFT
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.6_top_k_4
    # Qwen3-4B-SFT_hotpotqa_ep3_bsz48_th-1
    # Qwen3-8B-SFT_hotpotqa_ep3_bsz48_th-1
    # Qwen3-8B-SFT_hotpotqa_ep3_bsz48_lr5e-5_th-1

    # ----------------------------
    # Final Runs: GRPO
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak
    # Qwen3-4B-SFT_hotpotqa_ep3_bsz48_th-1-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak
    # Qwen3-8B-SFT_hotpotqa_ep3_bsz48_th-1-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak
    # Qwen3-8B-SFT_hotpotqa_ep3_bsz48_lr5e-5_th-1-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak

    # ----------------------------
    # Ablation: pre-filtering and curriculum
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.6_top_k_4-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-tier1_7-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.6_top_k_4-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-tier1_7-curric-nak

    # Ablation: em_size, em_size_4b,lr=1e-6
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-em_size-2ph-prsft-wcount-th0.6-topk4-nak
    # Qwen3-4B-SFT_hotpotqa_ep3_bsz48_th-1-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-em_size-2ph-prsft-wcount-th0.6-topk4-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-step500-n7000-f1-2ph-prcontext_only-wcount-th0.6-topk4-nak

    # # Ablation: early stopping, zero rl, 1-phase only, vanilla grpo
    # phase 1 final
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_1phase-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-th0.6-topk4-nak

    # Ablation: N16
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.6_top_k_4-grpo-tbs512-N16-K4-B32-M3-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak

    # Ablation: top1-th0.9
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.9_top_k_1
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.9_top_k_1-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.9-topk1-nak

    # Ablation: only 1-phase
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-th0.6-topk4-nak

    # Ablation: vanilla grpo
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-vanilla-th0.6-topk4-nak

    # ----------------------------
    # SFT Checkpoints
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1/checkpoint-92
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1/checkpoint-184
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1/checkpoint-276
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1/checkpoint-368
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1/checkpoint-735

    # GRPO on early sft
    # checkpoint-92-grpo-tbs512-N16-K4-B32-M3-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak
    # checkpoint-184-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak
    # checkpoint-276-grpo-tbs512-N16-K4-B32-M3-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak
    # checkpoint-368-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak

    # Zero RL
    # Qwen3-4B-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1_format-2ph-prformatted_zero_rl_v7-wcount-th0.6-topk4-nak/checkpoint-100
    # Qwen3-1.7B-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1_format-2ph-prformatted_zero_rl_v5-wcount-th0.6-topk4-nak/checkpoint-100
    # Qwen3-1.7B-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prformatted_zero_rl_v5-wcount-th0.6-topk4-nak

    # ----------------------------
)

# TWO_PHASE_METHOD="lmlm"
TWO_PHASE_METHOD="two_phase"

# ── Datasets and splits ──────────────────────────────────────────────────────
# NOTE: train_val1k is only valid for hotpotqa. musique/2wiki only support "dev".
#       mquake supports: eval-original, eval-edit, eval-edit-new, train
DATASET_LST=(
    # synthworlds
    # trivia_qa
    hotpotqa
    musique
    2wiki
    # mquake
)
SPLIT_LST=(
    # train_val1k
    dev
    # eval-original
    # eval-edit
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

            # Determine group key and job name based on SUBMIT_GRANULARITY:
            #   per_ckpt           — one job per checkpoint (all datasets combined)
            #   per_benchmark      — one job per dataset   (all checkpoints combined)
            #   per_ckpt_benchmark — one job per (checkpoint, dataset) [default]
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
            --partition=kilian \
            --gres=gpu:nvidia_rtx_a6000:1 \
            --requeue \
            "${TMP_SCRIPT}"
        rm -f "${TMP_SCRIPT}"
    else
        bash "${TMP_SCRIPT}"
        rm -f "${TMP_SCRIPT}"
    fi
done

echo "Evaluation complete!"