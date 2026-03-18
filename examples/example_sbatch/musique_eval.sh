#!/bin/bash
#SBATCH --job-name=lmlm_multihop_eval
#SBATCH --partition=jjs533,gpu-interactive
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -euo pipefail

# --------- user config ----------
SAVE_VERSION="v1"
MODEL_PATH="/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-4B-SFT_ep5_bsz48"

# --------- conda init ----------
source /home/as2637/miniconda/etc/profile.d/conda.sh
conda activate lmlm_multihop

# --------- env vars ----------
export MODEL_PATH="${MODEL_PATH}"

echo "Running eval with:"
echo "  SAVE_VERSION = ${SAVE_VERSION}"
echo "  MODEL_PATH   = ${MODEL_PATH}"
echo "  CONDA_ENV    = $(conda info --envs | grep '\*')"
echo "  CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"

nvidia-smi

# --------- run ----------
bash scripts/eval_lmlm_multihop.sh \
  --save_version "${SAVE_VERSION}"