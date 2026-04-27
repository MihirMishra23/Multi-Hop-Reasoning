#!/bin/bash
#SBATCH --mail-user=ryan.noonan@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=new-4b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --partition=jjs533
#SBATCH --constraint='6000ada'
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


echo "Starting SFT training..."
echo "Model size: ${MODEL_SIZE:-4B}"
if [ -n "${THRESHOLD:-}" ]; then
    echo "Retrieval threshold: $THRESHOLD"
    bash ./scripts/sft_train.sh --model_size ${MODEL_SIZE:-4B} --retrieval_threshold $THRESHOLD
else
    echo "No retrieval threshold specified"
    bash ./scripts/sft_train.sh --model_size ${MODEL_SIZE:-4B}
fi

echo "SFT complete!"
