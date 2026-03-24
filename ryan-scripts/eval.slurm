#!/bin/bash
#SBATCH --mail-user=ryan.noonan@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=grpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --constraint='6000ada'
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs


echo "Launching eval..."
bash /home/rtn27/Multi-Hop-Reasoning/ryan-scripts/eval_lmlm_multihop_grid_ry.sh
echo "Eval complete!"