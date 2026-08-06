#!/bin/bash
# Evaluate the development-selected IRCoT settings on the exact released
# 500-question test subsets with each of the three official prompt sets.

set -euo pipefail

: "${IRCOT_RETRIEVER_URL:?Set IRCOT_RETRIEVER_URL to the official service}"
SBATCH_BIN="${SBATCH_BIN:-/usr/local/slurm/current/bin/sbatch}"
RUN_TAG="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
IRCOT_ROOT="${IRCOT_ROOT:-/share/j_sun/rtn27/ircot_faithful}"
OUTPUT_DIR="${OUTPUT_DIR:-${IRCOT_ROOT}/predictions/test_best/${RUN_TAG}}"
PROMPT_SETS=(1 2 3)

for dataset in hotpotqa 2wiki musique; do
    for model in qwen3-1.7b qwen3-4b; do
        case "${dataset}/${model}" in
            hotpotqa/qwen3-1.7b) retrieval_k=8; distractors=1 ;;
            hotpotqa/qwen3-4b) retrieval_k=6; distractors=1 ;;
            2wiki/qwen3-1.7b) retrieval_k=4; distractors=3 ;;
            2wiki/qwen3-4b) retrieval_k=6; distractors=1 ;;
            musique/qwen3-1.7b) retrieval_k=4; distractors=2 ;;
            musique/qwen3-4b) retrieval_k=2; distractors=2 ;;
            *) echo "Missing selected setting for ${dataset}/${model}" >&2; exit 2 ;;
        esac
        case "${model}" in
            qwen3-1.7b)
                gpu_constraint="gpu-mid"
                min_gpu_memory_mib=0
                ;;
            qwen3-4b)
                gpu_constraint="${IRCOT_4B_GPU_CONSTRAINT:-gpu-high&a100}"
                min_gpu_memory_mib="${IRCOT_4B_MIN_GPU_MEMORY_MIB:-45000}"
                ;;
        esac
        for prompt_set in "${PROMPT_SETS[@]}"; do
            job_name="ircot_test_${dataset}_${model}_ps${prompt_set}"
            "${SBATCH_BIN}" \
                --job-name "${job_name}" \
                --time 16:00:00 \
                --constraint "${gpu_constraint}" \
                --export="ALL,DATASET=${dataset},MODEL=${model},RETRIEVAL_K=${retrieval_k},DISTRACTOR_COUNT=${distractors},PROMPT_SET=${prompt_set},EVALUATION_SPLIT=test,TOTAL_COUNT=500,SAVE_LABEL=papertest,RUN_TAG=${RUN_TAG},IRCOT_ROOT=${IRCOT_ROOT},OUTPUT_DIR=${OUTPUT_DIR},IRCOT_RETRIEVER_URL=${IRCOT_RETRIEVER_URL},MIN_GPU_MEMORY_MIB=${min_gpu_memory_mib}" \
                scripts/ircot_dev_grid_job.slurm
        done
    done
done
