#!/bin/bash
#SBATCH --mail-user=ryan.noonan@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=methods-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --partition=jjs533
#SBATCH --constraint='gpu-mid|gpu-high'
#SBATCH --time=10:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# Activate conda env from /scratch (faster than shared conda)
eval "$(conda shell.bash hook)"
conda activate /scratch/rtn27/multi-hop-reasoning
# conda activate multi-hop-reasoning

echo "Starting Eval..."


echo "Starting Eval..."

EVAL_MODE="latest"   # options: latest | all

# MODEL_ROOT="/share/j_sun/lz586/checkpoints/lmlm_multi_hop"
SFT_MODEL_ROOT=/share/j_sun/lz586/checkpoints/lmlm_multi_hop
GRPO_MODEL_ROOT=/share/j_sun/lmlm_multihop/checkpoints/debug
MODEL_NAMES=(
    Qwen3-4B
)

# Methods to evaluate
METHODS=(
    rag
    # react
    # Add other methods here as needed
)

DATASET_LST=(
    hotpotqa
    musique
    2wiki
    # mquake
)
SPLIT_lst=(
    # train_val1k
    dev
)
SAVE_VERSION=""
NUM_SAMPLES=1000
# USE_TRAIN_PARAMS=""
USE_TRAIN_PARAMS="--use-train-params"      # set to "--concat-all-db" to build unified database
OUTPUT_DIR="./rag-march-28-output"
SETTING="fullwiki"

# -----------------------
# Main loop
# -----------------------
for SPLIT in "${SPLIT_lst[@]}"; do

    for MODEL_NAME in "${MODEL_NAMES[@]}"; do

        for METHOD in "${METHODS[@]}"; do
            for DATASET in "${DATASET_LST[@]}"; do
                # Skip non-hotpotqa datasets when using train_val1k split
                if [[ "${SPLIT}" == "train_val1k" && "${DATASET}" != "hotpotqa" ]]; then
                    echo "Skipping ${DATASET} for ${SPLIT} split"
                    continue
                fi

                echo "========== Evaluating ${MODEL_NAME} on ${DATASET} with method ${METHOD} =========="
                bash /home/rtn27/Multi-Hop-Reasoning/ryan-scripts/eval_multihop.sh \
                    --model_path "${MODEL_NAME}" \
                    --dataset "${DATASET}" \
                    --split "${SPLIT}" \
                    --num_samples ${NUM_SAMPLES} \
                    --save_version "${SAVE_VERSION}" \
                    --method "${METHOD}" \
                    --output-dir "${OUTPUT_DIR}" \
                    --setting "${SETTING}"
            done
        done
    done
done

echo "Evaluation complete!"