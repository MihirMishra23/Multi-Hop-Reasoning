#!/bin/bash
#SBATCH --job-name=build_run_2wiki_db
#SBATCH --partition=jjs533,gpu-interactive
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/build_db_run%j.out
#SBATCH --error=logs/build_db_run%j.err

set -euo pipefail

# --- fix conda deactivate hook + nounset ---
export CONDA_BACKUP_CXX="${CONDA_BACKUP_CXX:-}"

# Load conda properly (guard against nounset issues in conda scripts)
set +u
source /home/as2637/miniconda/etc/profile.d/conda.sh
conda activate lmlm_multihop
set -u

# Export API keys (FYI: avoid hardcoding real keys in scripts)
export GEMINI_API_KEY=""

echo "Using Python: $(which python)"
nvidia-smi || true

bash scripts/eval_lmlm_multihop.sh