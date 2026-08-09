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
EVALUATION_SPLIT="${EVALUATION_SPLIT:-dev}"
PROMPT_SET="${PROMPT_SET:-1}"
RETRIEVAL_K="${RETRIEVAL_K:-6}"
DISTRACTOR_COUNT="${DISTRACTOR_COUNT:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-${IRCOT_ROOT}/predictions/official_smoke}"

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
case "${EVALUATION_SPLIT}" in dev|test) ;; *) echo "Unsupported EVALUATION_SPLIT=${EVALUATION_SPLIT}" >&2; exit 2 ;; esac
case "${PROMPT_SET}" in 1|2|3) ;; *) echo "Unsupported PROMPT_SET=${PROMPT_SET}" >&2; exit 2 ;; esac
case "${RETRIEVAL_K}" in 2|4|6|8) ;; *) echo "Unsupported RETRIEVAL_K=${RETRIEVAL_K}" >&2; exit 2 ;; esac
case "${DISTRACTOR_COUNT}" in 1|2|3) ;; *) echo "Unsupported DISTRACTOR_COUNT=${DISTRACTOR_COUNT}" >&2; exit 2 ;; esac

evaluation_file="${IRCOT_ROOT}/data/processed_data/${upstream_dataset}/${EVALUATION_SPLIT}_subsampled.jsonl"

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate multi-hop-reasoning
cd "${REPO_DIR}"

test -f "${evaluation_file}"
curl -fsS "${IRCOT_RETRIEVER_URL}/" >/dev/null

python src/eval_multihop.py \
    --method ircot \
    --dataset "${DATASET}" \
    --setting distractor \
    --split "${EVALUATION_SPLIT}" \
    --model "${MODEL}" \
    --model-revision "${model_revision}" \
    --batch-size 1 \
    --total-count 2 \
    --output-dir "${OUTPUT_DIR}" \
    --save-version "_official_${EVALUATION_SPLIT}_ps${PROMPT_SET}_${SLURM_JOB_ID}" \
    --save-every 1 \
    --temperature 0 \
    --max-tokens 300 \
    --ircot-retriever-url "${IRCOT_RETRIEVER_URL}" \
    --ircot-index-manifest "${IRCOT_ROOT}/manifests/${upstream_dataset}_index.json" \
    --ircot-evaluation-file "${evaluation_file}" \
    --ircot-prompt-dir provenance/ircot/prompts \
    --ircot-prompt-set "${PROMPT_SET}" \
    --ircot-retrieval-k "${RETRIEVAL_K}" \
    --ircot-distractor-count "${DISTRACTOR_COUNT}" \
    --ircot-max-evidence 15 \
    --ircot-max-steps 10 \
    --ircot-generator-max-tokens 300 \
    --debug-evidence
