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

source /share/apps/software/anaconda3/etc/profile.d/conda.sh

# Export API keys (FYI: avoid hardcoding real keys in scripts)
export GEMINI_API_KEY=""

echo "Using Python: $(which python)"
nvidia-smi || true

bash scripts/eval_lmlm_multihop.sh
