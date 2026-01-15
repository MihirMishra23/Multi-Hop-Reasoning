#!/bin/bash
#SBATCH --mail-user=ryan.noonan@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=grpo
#SBATCH --nodes=1
#SBATCH -p jjs533
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --constraint='h100'
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err


set -euo pipefail

# Run from the directory where you submitted the job
cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs
source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate multi-hop-reasoning


echo "Starting GRPO..."
python scripts/rl_test.py

echo "GRPO complete!"
