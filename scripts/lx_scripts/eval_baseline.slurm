#!/bin/bash


# SLURM Directives
#SBATCH -J baseline_eval                            # Job name
#SBATCH -o /home/lz586/icl/slurm_output/lmlm_multi_hop/baseline_eval_%j.out   # Standard output file
#SBATCH -e /home/lz586/icl/slurm_output/lmlm_multi_hop/baseline_eval_%j.err   # Standard error file
#SBATCH -N 1                                    # Number of nodes
#SBATCH -n 1                                    # Number of tasks (or cores)
#SBATCH --get-user-env                          # Retrieve the user's login environment
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G                              # Memory requested
#SBATCH -t infinite                              # Time limit
#SBATCH --partition=aimi,kilian,jjs533,default_partition          # Request partition
#SBATCH --gres=gpu:nvidia_rtx_a6000:1                 # Request 1 NVIDIA A6000 GPU


# Load the required environment
source activate mem

# Navigate to the working directory
cd /home/lz586/icl/Multi-Hop-Reasoning/


echo "Starting Baseline Eval..."

# Models to evaluate
MODEL_LST=(
    qwen3-1.7b
    qwen3-4b    # NOTE: add "qwen3-4b": "Qwen/Qwen3-4B" to MODEL_MAP in src/llm/qwen.py first
)

# Methods: direct (zero-shot), icl (in-context learning / CoT), rag
METHOD_LST=(
    # direct
    # icl
    rag
    # cot
)

DATASET_LST=(
    hotpotqa
    musique
    2wiki
)
SPLIT_LST=(
    dev
)
NUM_SAMPLES=1000

# -----------------------
# Main loop
# -----------------------
for SPLIT in "${SPLIT_LST[@]}"; do
    for MODEL in "${MODEL_LST[@]}"; do
        for METHOD in "${METHOD_LST[@]}"; do
            for DATASET in "${DATASET_LST[@]}"; do
                echo "========== Evaluating ${MODEL} | ${METHOD} | ${DATASET} | ${SPLIT} =========="
                bash /home/lz586/icl/Multi-Hop-Reasoning/scripts/eval_lmlm_multihop.sh \
                    --model_path "unused" \
                    --method "${METHOD}" \
                    --llm_model "${MODEL}" \
                    --dataset "${DATASET}" \
                    --split "${SPLIT}" \
                    --num_samples ${NUM_SAMPLES}
            done
        done
    done
done

echo "Baseline evaluation complete!"
