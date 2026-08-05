#!/bin/bash
# Two released examples through the real paper corpus/index before the 100-row grid.

#SBATCH --job-name=ircot_official_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=default_partition
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-mid
#SBATCH --output=/home/rtn27/ircot_official_smoke_%j.out
#SBATCH --error=/home/rtn27/ircot_official_smoke_%j.err

set -euo pipefail
: "${IRCOT_RETRIEVER_URL:?Set IRCOT_RETRIEVER_URL}"

REPO_DIR="${REPO_DIR:-/home/rtn27/Multi-Hop-Reasoning-ircot-table3}"
IRCOT_ROOT="${IRCOT_ROOT:-/share/j_sun/rtn27/ircot_faithful}"
DATASET="${DATASET:-2wiki}"
MODEL="${MODEL:-qwen3-1.7b}"

case "${DATASET}" in
    hotpotqa) upstream_dataset=hotpotqa ;;
    2wiki) upstream_dataset=2wikimultihopqa ;;
    musique) upstream_dataset=musique ;;
    *) echo "Unsupported DATASET=${DATASET}" >&2; exit 2 ;;
esac
case "${MODEL}" in
    qwen3-1.7b) model_revision=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e ;;
    qwen3-4b) model_revision=1cfa9a7208912126459214e8b04321603b3df60c ;;
    *) echo "Unsupported MODEL=${MODEL}" >&2; exit 2 ;;
esac

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate multi-hop-reasoning
cd "${REPO_DIR}"

python src/eval_multihop.py \
    --method ircot \
    --dataset "${DATASET}" \
    --setting distractor \
    --split dev \
    --model "${MODEL}" \
    --model-revision "${model_revision}" \
    --batch-size 1 \
    --total-count 2 \
    --output-dir "${IRCOT_ROOT}/predictions/official_smoke" \
    --save-version "_official_${SLURM_JOB_ID}" \
    --save-every 1 \
    --temperature 0 \
    --max-tokens 300 \
    --ircot-retriever-url "${IRCOT_RETRIEVER_URL}" \
    --ircot-index-manifest "${IRCOT_ROOT}/manifests/${upstream_dataset}_index.json" \
    --ircot-evaluation-file "${IRCOT_ROOT}/data/processed_data/${upstream_dataset}/dev_subsampled.jsonl" \
    --ircot-prompt-dir provenance/ircot/prompts \
    --ircot-prompt-set 1 \
    --ircot-retrieval-k 6 \
    --ircot-distractor-count 2 \
    --ircot-max-evidence 15 \
    --ircot-max-steps 10 \
    --ircot-generator-max-tokens 300 \
    --debug-evidence
