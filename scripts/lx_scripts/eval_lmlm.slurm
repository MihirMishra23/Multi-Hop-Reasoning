#!/bin/bash


# SLURM Directives
#SBATCH -J lmlm_eval                            # Job name
#SBATCH -o /home/lz586/icl/slurm_output/lmlm_multi_hop/lmlm_eval_%j.out   # Standard output file
#SBATCH -e /home/lz586/icl/slurm_output/lmlm_multi_hop/lmlm_eval_%j.err   # Standard error file
#SBATCH -N 1                                    # Number of nodes
#SBATCH -n 1                                    # Number of tasks (or cores)
#SBATCH --get-user-env                          # Retrieve the user's login environment
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G                              # Memory requested (2GB per node)
#SBATCH -t infinite                              # Time limit (2 hours)
#SBATCH --partition=aimi,kilian,jjs533,default_partition          # Request partition
#SBATCH --gres=gpu:nvidia_rtx_a6000:1                 # Request 1 NVIDIA A6000 GPU nvidia_rtx_a6000


# Load the required environmentt
source activate mem

# Navigate to the working directory
cd /home/lz586/icl/Multi-Hop-Reasoning/


echo "Starting Eval..."

EVAL_MODE="latest"   # options: latest | all

SFT_MODEL_ROOT2="/share/j_sun/lz586/checkpoints/lmlm_multi_hop"
SFT_MODEL_ROOT=/share/j_sun/rtn27/checkpoints/lmlm_multi_hop
GRPO_MODEL_ROOT=/share/j_sun/lmlm_multihop/checkpoints/debug/
MODEL_PATH_LST=(
    # ----------------------------
    # Main CKPTs on hotpotqa
    # Qwen3-1.7B-SFT_ep5_bsz48
    # Qwen3-1.7B-SFT_ep5_bsz48-grpo-g8-bs16-s8-b0.0-ep3-n7000
    # Qwen3-1.7B-SFT_ep5_bsz48-grpo-g8-bs16-s8-b0.0-ep5-n8000
    # # Qwen3-4B-SFT_ep5_bsz48
    # Qwen3-4B-SFT_ep5_bsz48-grpo-g8-bs16-s8-b0.0-ep5-n8000
    # ----------------------------

    # Ablation: F1 threshold (quality threshold of gemini rollouts)
    # Qwen3-1.7B-SFT_ep5_bsz48_th0.8-grpo-g8-bs16-s8-b0.0-ep5-n8000
    # Qwen3-1.7B-SFT_ep5_bsz48_th-2-grpo-g8-bs16-s8-b0.0-ep5-n8000
    # Qwen3-1.7B-SFT_ep5_bsz48_th0.3-grpo-g8-bs16-s8-b0.0-ep5-n8000

    # Debug
    # Qwen3-1.7B-SFT_mquake-cf-debug_ep5_bsz48_th1.0
    # Qwen3-1.7B-SFT_mquake-real-debug_ep5_bsz48_th1.0
    
    # Ablation: base model version
    # Qwen2.5-3B-SFT_ep5_bsz48_th-1

    # Ablation: new db format. return triples
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

    # New 2 phase models
    # SFT
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k
    # Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed
    
    # full run
    # Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed-grpo-B1-K4-M4-bs16-s8-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak

    # # n1000
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B1-K4-M4-bs16-s16-b0.0-ep5-n1000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B2-K4-M2-bs16-s8-b0.0-ep5-n1000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak

    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B0-K4-M8-bs16-s8-b0.0-ep5-n1000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B1-K8-M2-bs16-s8-b0.0-ep5-n1000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B8-K4-M8-bs16-s16-b0.0-ep5-n1000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak


    # # n500
    # Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed-grpo-B2-K4-M2-bs16-s4-b0.0-ep10-n500-f1-v2-p1binary-ptcontext_only-wmcount_dynamic-th0.6-topk4-debug-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed-grpo-B2-K4-M2-bs16-s4-b0.0-ep10-n500-em_size-v2-p1binary-ptcontext_only-wmcount_dynamic-th0.6-topk4-debug-nak

    # # new full run
    # huge performance drop
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N8-K4-B64-M2-b0.0-step500-n7000-f1-2ph-prcontext_only-wcount_dynamic-th0.6-topk4-nak
    
    # the best one
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-step500-n7000-f1-2ph-prcontext_only-wcount-th0.6-topk4-nak
    
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-step500-n7000-em-2ph-prcontext_only-wcount-th0.6-topk4-nak
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs1024-N32-K4-B32-M7-b0.0-step500-n7000-em-2ph-prcontext_only-wcount-th0.6-topk4-nak

    # new full run v2
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-step500-n7000-f1-v2-2ph-prcontext_only-wcount-th0.6-topk4-nak

    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B32-K2-M4-bs16-s16-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak

    # Menghan
#     Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B32-K4-M8-bs16-s64-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak
# Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-step500-n7000-f1-v2-2ph-prcontext_only-wcount-th0.6-topk4-nak
# Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B16-K4-M8-bs16-s32-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak
# Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B32-K4-M8-bs16-s64-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak
# Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B64-K2-M4-bs16-s32-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak
# Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B128-K2-M4-bs16-s64-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak
# Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B16-K4-M8-bs16-s32-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak
# Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B128-K2-M4-bs16-s64-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak
# Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B32-K4-M8-bs16-s64-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak
# Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B16-K8-M4-bs16-s32-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak
# Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B8-K4-M8-bs16-s16-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak
# Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B32-K4-M4-bs16-s32-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak
# Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-B32-K2-M4-bs16-s16-b0.0-ep5-n7000-em_size-v2-p1binary-ptcontext_only-wmcount-th0.6-topk4-nak

# Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-step500-n7000-f1-v2-2ph-prcontext_only-wcount-th0.6-topk4-nak


    # # Ablation: 
    # # early stopping
    # checkpoint-368-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak
    # # zero rl format v5
    # Qwen3-1.7B-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1_format-2ph-prformatted_zero_rl_v5-wcount-th0.6-topk4-nak
    # # only phase 1
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak

    # SFT
    Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k
    Qwen3-4B-SFT_hotpotqa_ep3_bsz48_th-1

    # full run Main
    Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak
    Qwen3-4B-SFT_hotpotqa_ep3_bsz48_th-1-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak


    ##### Ablation: 
    # 2phase em_size
    Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-em_size-2ph-prsft-wcount-th0.6-topk4-nak
    
    # 2phase lr=1e-6
    Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-step500-n7000-f1-2ph-prcontext_only-wcount-th0.6-topk4-nak

    # 1 phase only other 2 variants
    Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-step500-n7000-f1-th0.6-topk4-nak
    Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-th0.6-topk4-nak
    # 1 phase only final
    Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-th0.6-topk4-nak

    # early sft
    checkpoint-368-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak

    # vanilla grpo
    Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-vanilla-th0.6-topk4-nak

    # zero rl
    Qwen3-4B-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1_format-2ph-prformatted_zero_rl_v7-wcount-th0.6-topk4-nak
    Qwen3-1.7B-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1_format-2ph-prformatted_zero_rl_v5-wcount-th0.6-topk4-nak
    Qwen3-1.7B-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prformatted_zero_rl_v5-wcount-th0.6-topk4-nak




)

TWO_PHASE_METHOD="two_phase"

DATASET_LST=(
    # hotpotqa
    # musique
    # 2wiki
    mquake
)
SPLIT_lst=(
    # train_val1k
    dev
)
SAVE_VERSION="_0325"
NUM_SAMPLES=1000
# USE_TRAIN_PARAMS=""
USE_TRAIN_PARAMS="--use-train-params"


# -----------------------
# Main loop
# -----------------------
for SPLIT in "${SPLIT_lst[@]}"; do
    for MODEL_DIR in "${MODEL_PATH_LST[@]}"; do

    # if no grpo in model dir, then use sft model root
    if [[ "${MODEL_DIR}" != *"grpo"* ]]; then
    # 1.7b SFT_MODEL_ROOT
    # others SFT_MODEL_ROOT2
        if [[ "${MODEL_DIR}" == *"1.7b"* ]]; then
            MODEL_ROOT="${SFT_MODEL_ROOT}"
        else
            MODEL_ROOT="${SFT_MODEL_ROOT2}"
        fi
    else
        MODEL_ROOT="${GRPO_MODEL_ROOT}"
    fi

    # Collect checkpoint dirs
    mapfile -t CKPT_DIRS < <(
        find "${MODEL_ROOT}/${MODEL_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
        | sort -V
    )

    if [[ ${#CKPT_DIRS[@]} -eq 0 ]]; then
        echo "No checkpoint-* directories found under: ${MODEL_ROOT}/${MODEL_DIR}" >&2
        continue
    fi

    if [[ "${EVAL_MODE}" == "latest" ]]; then
        CKPT_DIRS=( "${CKPT_DIRS[-1]}" )
    fi

    for CKPT in "${CKPT_DIRS[@]}"; do
        MODEL_PATH="${MODEL_ROOT}/${MODEL_DIR}/${CKPT}"

        for DATASET in "${DATASET_LST[@]}"; do
            echo "========== Evaluating ${MODEL_PATH} on ${DATASET} =========="
            bash /home/lz586/icl/Multi-Hop-Reasoning/scripts/eval_lmlm_multihop.sh \
                --model_path "${MODEL_PATH}" \
                --dataset "${DATASET}" \
                --split "${SPLIT}" \
                --num_samples ${NUM_SAMPLES} \
                --save_version "${SAVE_VERSION}" \
                --method "${TWO_PHASE_METHOD}" \
                ${USE_TRAIN_PARAMS}
            done
        done
    done
done

echo "Evaluation complete!"