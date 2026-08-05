#!/bin/bash
# Submit the paper's complete prompt-set-1 development grid: 100 released
# examples × k={2,4,6,8} × distractors={1,2,3} for each agreed Qwen model.

set -euo pipefail

: "${IRCOT_RETRIEVER_URL:?Set IRCOT_RETRIEVER_URL to the official service}"
SBATCH_BIN="${SBATCH_BIN:-/usr/local/slurm/current/bin/sbatch}"
RUN_TAG="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
DATASETS=(hotpotqa 2wiki musique)
MODELS=(qwen3-1.7b qwen3-4b)
RETRIEVAL_COUNTS=(2 4 6 8)
DISTRACTOR_COUNTS=(1 2 3)

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        case "${model}" in
            qwen3-1.7b) gpu_constraint="gpu-mid" ;;
            # MuSiQue can reach the full 8k-token IRCoT prompt limit.  Qwen3-4B
            # exhausts 24 GiB devices on those examples, so use 48 GiB Ampere
            # devices (or larger memory devices with the same features).
            qwen3-4b) gpu_constraint="gpu-high&ampere" ;;
            *) echo "Unsupported MODEL=${model}" >&2; exit 2 ;;
        esac
        for retrieval_k in "${RETRIEVAL_COUNTS[@]}"; do
            for distractors in "${DISTRACTOR_COUNTS[@]}"; do
                job_name="ircot_${dataset}_${model}_k${retrieval_k}_d${distractors}"
                "${SBATCH_BIN}" \
                    --job-name "${job_name}" \
                    --constraint "${gpu_constraint}" \
                    --export="ALL,DATASET=${dataset},MODEL=${model},RETRIEVAL_K=${retrieval_k},DISTRACTOR_COUNT=${distractors},RUN_TAG=${RUN_TAG},IRCOT_RETRIEVER_URL=${IRCOT_RETRIEVER_URL}" \
                    scripts/ircot_dev_grid_job.slurm
            done
        done
    done
done
