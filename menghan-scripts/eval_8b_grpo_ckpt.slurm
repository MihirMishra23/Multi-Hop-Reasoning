#!/bin/bash


# SLURM Directives
#SBATCH -J eval-8b-grpo
#SBATCH -o /home/mx253/slurm_output/lmlm_multihop/eval_8b_grpo_ckpt_%j.out
#SBATCH -e /home/mx253/slurm_output/lmlm_multihop/eval_8b_grpo_ckpt_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --partition=aimi,default_partition
#SBATCH --exclude=jjs533-compute-01,jjs533-compute-02,jjs533-compute-03
#SBATCH --gres=gpu:nvidia_rtx_a6000:1


eval "$(conda shell.bash hook)"
conda activate lmlm_b200

cd /share/j_sun/mx253/Multi-Hop-Reasoning


echo "Starting Eval..."
EVAL_MODE="all"   # "latest" = only the highest-step checkpoint; "all" = every checkpoint

MODEL_PATH_LST=(
    /share/j_sun/lmlm_multihop/checkpoints/main/Qwen3-8B-SFT_hotpotqa_ep3_bsz48_lr5e-5_th-1-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak
)

DATASET_LST=(
    hotpotqa
)


# -----------------------
# Main loop
# -----------------------
for MODEL_DIR in "${MODEL_PATH_LST[@]}"; do
    echo "Processing model dir: ${MODEL_DIR}"

    # Collect checkpoint dirs
    mapfile -t CKPT_DIRS < <(
        find "${MODEL_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
        | sort -V
    )

    if [[ ${#CKPT_DIRS[@]} -eq 0 ]]; then
        echo "No checkpoint-* directories found under: ${MODEL_DIR}" >&2
        continue
    fi

    if [[ "${EVAL_MODE}" == "latest" ]]; then
        CKPT_DIRS=( "${CKPT_DIRS[-1]}" )
    fi

    for CKPT in "${CKPT_DIRS[@]}"; do
        MODEL_PATH="${MODEL_DIR}/${CKPT}"

        for DATASET in "${DATASET_LST[@]}"; do
            echo "Evaluating ${MODEL_PATH} on ${DATASET}"
            bash ./scripts/eval_lmlm_multihop.sh \
                --model_path "${MODEL_PATH}" \
                --dataset "${DATASET}" \
                --method two_phase \
                --use-contexts all \
                --concat-all-db \
                --use-inverses \
                --num_samples 300
        done
    done
done

echo "Evaluation complete!"
