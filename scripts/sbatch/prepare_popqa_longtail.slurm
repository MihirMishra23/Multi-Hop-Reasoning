#!/bin/bash
#SBATCH --job-name=popqa1399_prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=jjs533
#SBATCH --time=00:30:00
#SBATCH --output=/home/rtn27/popqa1399_prepare_%j.out
#SBATCH --error=/home/rtn27/popqa1399_prepare_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/rtn27/Multi-Hop-Reasoning-popqa-longtail}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/share/j_sun/rtn27/popqa_longtail_1399}"

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/rtn27/multi-hop-reasoning
cd "${REPO_DIR}"

python scripts/prepare_popqa_longtail.py \
    --output-dir "${ARTIFACT_DIR}" \
    --seed 42
