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
#SBATCH --partition=kilian,jjs533,default_partition          # Request partition
# SBATCH --gres=gpu:nvidia_rtx_a6000:1                 # Request 1 NVIDIA A6000 GPU


# Load the required environmentt
source activate mem

# Navigate to the working directory
cd /home/lz586/icl/Multi-Hop-Reasoning/


echo "Starting Eval..."
MODEL_PATH_LST=(
    /share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-4B-SFT_ep5_bsz48-grpo-g8-bs16-s8-b0.0-ep5-n8000
    /share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_ep5_bsz48-grpo-g8-bs16-s8-b0.0-ep5-n8000
    /share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_ep5_bsz48_th0.8-grpo-g8-bs16-s8-b0.0-ep5-n8000
    /share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_ep5_bsz48_th-2-grpo-g8-bs16-s8-b0.0-ep5-n8000
    /share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_ep5_bsz48_th0.3-grpo-g8-bs16-s8-b0.0-ep5-n8000
    /share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-4B-SFT_ep5_bsz48
    /share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_ep5_bsz48
)

DATASET_LST=(
    # hotpotqa
    musique
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
                --dataset "${DATASET}"
        done
    done
done

echo "Evaluation complete!"
