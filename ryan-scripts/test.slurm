#!/bin/bash
#SBATCH --job-name=grpo
#SBATCH --nodes=1
#SBATCH -p jjs533
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --constraint='h100'
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euxo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

echo "PWD=$(pwd)"
echo "HOSTNAME=$(hostname)"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}"

ls -ld /scratch/rtn27/multi-hop-reasoning || true
ls -l /scratch/rtn27/multi-hop-reasoning/bin/activate || true
ls -l /home/rtn27/Multi-Hop-Reasoning/ryan-scripts/eval_lmlm_multihop_grid_ry.sh || true

source /scratch/rtn27/multi-hop-reasoning/bin/activate

echo "Launching eval..."
bash -x /home/rtn27/Multi-Hop-Reasoning/ryan-scripts/eval_lmlm_multihop_grid_ry.sh
echo "Eval complete!"