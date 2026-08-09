#!/bin/bash
set -euo pipefail

RUN_TAG="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
REPO_DIR="${REPO_DIR:-/home/rtn27/Multi-Hop-Reasoning-popqa-longtail}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/share/j_sun/rtn27/popqa_longtail_1399}"
OUTPUT_DIR="${OUTPUT_DIR:-/share/j_sun/rtn27/popqa_longtail_eval/${RUN_TAG}}"

prepare_job=$(sbatch --parsable \
    --export="ALL,REPO_DIR=${REPO_DIR},ARTIFACT_DIR=${ARTIFACT_DIR}" \
    scripts/sbatch/prepare_popqa_longtail.slurm)
echo "prepare ${prepare_job}"

submit_eval() {
    local method="$1" model_key="$2"
    local job_name="popqa1399_${method}_${model_key}"
    local job_id
    job_id=$(sbatch --parsable \
        --dependency="afterok:${prepare_job}" \
        --job-name="${job_name}" \
        --export="ALL,METHOD=${method},MODEL_KEY=${model_key},RUN_TAG=${RUN_TAG},REPO_DIR=${REPO_DIR},ARTIFACT_DIR=${ARTIFACT_DIR},OUTPUT_DIR=${OUTPUT_DIR}" \
        scripts/sbatch/eval_popqa_longtail.slurm)
    echo "${method} ${model_key} ${job_id}"
}

# Submit the expensive KBEVO jobs first so they take the first free accelerators.
submit_eval two_phase kbevo-1.7b
submit_eval two_phase kbevo-4b
submit_eval rag qwen3-1.7b
submit_eval rag qwen3-4b
submit_eval direct qwen3-1.7b
submit_eval direct qwen3-4b
