#!/bin/bash
#SBATCH --mail-user=ryan.noonan@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=32-trivia-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --partition=jjs533
#SBATCH --constraint='gpu-mid|gpu-high'
#SBATCH --time=8:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# Activate conda env from /scratch (faster than shared conda)
eval "$(conda shell.bash hook)"
if [[ "${SLURM_JOB_PARTITION}" == "jjs533" ]]; then
    conda activate /scratch/rtn27/multi-hop-reasoning
else
    conda activate multi-hop-reasoning
fi

echo "Starting Eval..."


echo "Starting Eval..."

EVAL_MODE="latest"   # options: latest | all

MODEL_ROOT="/share/j_sun/lz586/checkpoints/lmlm_multi_hop"
# SFT_MODEL_ROOT=/share/j_sun/lmlm_multihop/checkpoints/main/
GRPO_MODEL_ROOT=/share/j_sun/lmlm_multihop/checkpoints/debug
MODEL_PATH_LST=(
    # SFT models
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.6_top_k_4
    /share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-4B-SFT_hotpotqa_ep3_bsz48_th-1
    # main 2 phase GRPO runs
    # Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak
    # Qwen3-4B-SFT_hotpotqa_ep3_bsz48_th-1-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak
)

TWO_PHASE_METHOD="two_phase"

DATASET_LST=(
    # synthworlds
    popqa
    # hotpotqa
    # musique
    # 2wiki
    trivia_qa
    # mquake
)
SPLIT_lst=(
    # train_val1k
    dev
)
SAVE_VERSION="concat-all"
NUM_SAMPLES=1000
# USE_TRAIN_PARAMS=""
USE_TRAIN_PARAMS="--use-train-params"
CONCAT_ALL_DB=""
CONCAT_ALL_DB="--concat-all-db"              # set to "--concat-all-db" to build unified database
USE_CONTEXTS="all"         # options: "golden" | "all"
OUTPUT_DIR="./march-31-trivia-qa-pop-qa-final"
SETTING="distractor"

# -----------------------
# Main loop
# -----------------------
for SPLIT in "${SPLIT_lst[@]}"; do
    for MODEL_DIR in "${MODEL_PATH_LST[@]}"; do

    # if no grpo in model dir, then use sft model root
    if [[ "${MODEL_DIR}" != *"grpo"* ]]; then
        MODEL_ROOT="${SFT_MODEL_ROOT}"
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
            # Skip non-hotpotqa datasets when using train_val1k split
            if [[ "${SPLIT}" == "train_val1k" && "${DATASET}" != "hotpotqa" ]]; then
                echo "Skipping ${DATASET} for ${SPLIT} split"
                continue
            fi

            # Automatically set setting to 'qa-sm' for synthworlds dataset
            if [[ "${DATASET}" == "synthworlds" ]]; then
                DATASET_SETTING="qa-sm"
                DATASET_USE_CONTEXTS="golden"
                echo "NOTE: Automatically setting --use-contexts=golden for synthworlds dataset (only golden contexts are available)"
            else
                DATASET_SETTING="${SETTING}"
                DATASET_USE_CONTEXTS="${USE_CONTEXTS}"
            fi

            echo "========== Evaluating ${MODEL_PATH} on ${DATASET} (setting=${DATASET_SETTING}, use-contexts=${DATASET_USE_CONTEXTS}) =========="
            bash /home/rtn27/Multi-Hop-Reasoning/ryan-scripts/eval_multihop.sh \
                --model_path "${MODEL_PATH}" \
                --dataset "${DATASET}" \
                --split "${SPLIT}" \
                --num_samples ${NUM_SAMPLES} \
                --save_version "${SAVE_VERSION}" \
                --method "${TWO_PHASE_METHOD}" \
                --output-dir "${OUTPUT_DIR}" \
                ${USE_TRAIN_PARAMS} \
                ${CONCAT_ALL_DB} \
                --use-contexts "${DATASET_USE_CONTEXTS}" \
                --setting "${DATASET_SETTING}"
            done
        done
    done
done

echo "Evaluation complete!"