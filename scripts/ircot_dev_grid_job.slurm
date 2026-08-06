#!/bin/bash
# One cell of the exact IRCoT development sweep or best-HP test evaluation.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=default_partition
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-mid
#SBATCH --requeue
#SBATCH --output=/home/rtn27/ircot_dev_%x_%j.out
#SBATCH --error=/home/rtn27/ircot_dev_%x_%j.err

set -euo pipefail

# The launcher overrides the default constraint for Qwen3-4B so that the
# longest official prompts run on >=48 GiB Ampere devices.

: "${DATASET:?Set DATASET to hotpotqa, 2wiki, or musique}"
: "${MODEL:?Set MODEL to qwen3-1.7b or qwen3-4b}"
: "${IRCOT_RETRIEVER_URL:?Set IRCOT_RETRIEVER_URL}"
: "${RETRIEVAL_K:?Set RETRIEVAL_K}"
: "${DISTRACTOR_COUNT:?Set DISTRACTOR_COUNT}"
: "${RUN_TAG:?Set RUN_TAG}"

REPO_DIR="${REPO_DIR:-/home/rtn27/Multi-Hop-Reasoning-ircot-table3}"
IRCOT_ROOT="${IRCOT_ROOT:-/share/j_sun/rtn27/ircot_faithful}"
OUTPUT_DIR="${OUTPUT_DIR:-${IRCOT_ROOT}/predictions/dev_grid}"
EVALUATION_SPLIT="${EVALUATION_SPLIT:-dev}"
TOTAL_COUNT="${TOTAL_COUNT:-100}"
PROMPT_SET="${PROMPT_SET:-1}"
SAVE_LABEL="${SAVE_LABEL:-paperdev}"

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
case "${RETRIEVAL_K}" in 2|4|6|8) ;; *) echo "Unsupported RETRIEVAL_K=${RETRIEVAL_K}" >&2; exit 2 ;; esac
case "${DISTRACTOR_COUNT}" in 1|2|3) ;; *) echo "Unsupported DISTRACTOR_COUNT=${DISTRACTOR_COUNT}" >&2; exit 2 ;; esac
case "${EVALUATION_SPLIT}" in dev|test) ;; *) echo "Unsupported EVALUATION_SPLIT=${EVALUATION_SPLIT}" >&2; exit 2 ;; esac
case "${PROMPT_SET}" in 1|2|3) ;; *) echo "Unsupported PROMPT_SET=${PROMPT_SET}" >&2; exit 2 ;; esac
case "${TOTAL_COUNT}" in 100|500) ;; *) echo "Unsupported TOTAL_COUNT=${TOTAL_COUNT}" >&2; exit 2 ;; esac

evaluation_file="${IRCOT_ROOT}/data/processed_data/${upstream_dataset}/${EVALUATION_SPLIT}_subsampled.jsonl"

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate multi-hop-reasoning
cd "${REPO_DIR}"

test -f "${evaluation_file}"
test -f "${IRCOT_ROOT}/manifests/${upstream_dataset}_index.json"
curl -fsS "${IRCOT_RETRIEVER_URL}/" >/dev/null

echo "code_commit=$(git rev-parse HEAD)"
echo "dataset=${DATASET} model=${MODEL} split=${EVALUATION_SPLIT} count=${TOTAL_COUNT} k=${RETRIEVAL_K} distractors=${DISTRACTOR_COUNT} prompt_set=${PROMPT_SET}"
echo "evaluation_file=${evaluation_file}"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader

MIN_GPU_MEMORY_MIB="${MIN_GPU_MEMORY_MIB:-0}"
gpu_memory_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
if ! [[ "${MIN_GPU_MEMORY_MIB}" =~ ^[0-9]+$ && "${gpu_memory_mib}" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU memory check: required=${MIN_GPU_MEMORY_MIB} MiB assigned=${gpu_memory_mib} MiB" >&2
    exit 2
fi
if (( gpu_memory_mib < MIN_GPU_MEMORY_MIB )); then
    echo "Assigned GPU has ${gpu_memory_mib} MiB; this job requires at least ${MIN_GPU_MEMORY_MIB} MiB" >&2
    exit 78
fi

python src/eval_multihop.py \
    --method ircot \
    --dataset "${DATASET}" \
    --setting distractor \
    --split "${EVALUATION_SPLIT}" \
    --model "${MODEL}" \
    --model-revision "${model_revision}" \
    --batch-size 1 \
    --start-index 0 \
    --total-count "${TOTAL_COUNT}" \
    --save-every 10 \
    --resume \
    --output-dir "${OUTPUT_DIR}" \
    --save-version "_${SAVE_LABEL}_ps${PROMPT_SET}_${RUN_TAG}" \
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
