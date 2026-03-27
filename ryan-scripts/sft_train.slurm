#!/bin/bash
#SBATCH --mail-user=ryan.noonan@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=sft_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --constraint='6000ada'
#SBATCH --time=10:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# Activate conda env from /scratch (faster than shared conda)
eval "$(conda shell.bash hook)"
#conda activate /scratch/rtn27/multi-hop-reasoning
conda activate multi-hop-reasoning

echo "launching SFT"

MODEL_SIZES=(4B 8B)

for MODEL_SIZE in "${MODEL_SIZES[@]}"; do
    echo "========== Training model size: ${MODEL_SIZE} ==========="
    bash /home/rtn27/Multi-Hop-Reasoning/scripts/sft_train.sh --model_size "${MODEL_SIZE}"
done

echo "SFT training complete for all model sizes!"

EVAL_MODE="latest"   # options: latest | all