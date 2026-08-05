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
        for retrieval_k in "${RETRIEVAL_COUNTS[@]}"; do
            for distractors in "${DISTRACTOR_COUNTS[@]}"; do
                job_name="ircot_${dataset}_${model}_k${retrieval_k}_d${distractors}"
                "${SBATCH_BIN}" \
                    --job-name "${job_name}" \
                    --export="ALL,DATASET=${dataset},MODEL=${model},RETRIEVAL_K=${retrieval_k},DISTRACTOR_COUNT=${distractors},RUN_TAG=${RUN_TAG},IRCOT_RETRIEVER_URL=${IRCOT_RETRIEVER_URL}" \
                    scripts/ircot_dev_grid_job.slurm
            done
        done
    done
done
