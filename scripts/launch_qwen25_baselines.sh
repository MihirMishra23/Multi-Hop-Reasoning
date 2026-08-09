#!/bin/bash
set -euo pipefail

RUN_TAG="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
POPQA_PREP_JOB="${2:-}"
REPO_DIR="${REPO_DIR:-/home/rtn27/Multi-Hop-Reasoning-qwen25-table3}"
OUTPUT_DIR="${OUTPUT_DIR:-/share/j_sun/rtn27/qwen25_table3_eval/${RUN_TAG}}"

submit() {
    local method="$1" dataset="$2"
    local dependency=()
    if [[ "${dataset}" == "popqa" && -n "${POPQA_PREP_JOB}" ]]; then
        dependency=(--dependency="afterok:${POPQA_PREP_JOB}")
    fi
    local job_id
    job_id=$(sbatch --parsable \
        "${dependency[@]}" \
        --job-name="q25_${method}_${dataset}" \
        --export="ALL,METHOD=${method},DATASET=${dataset},RUN_TAG=${RUN_TAG},REPO_DIR=${REPO_DIR},OUTPUT_DIR=${OUTPUT_DIR}" \
        scripts/sbatch/eval_qwen25_baseline.slurm)
    echo "${method} ${dataset} ${job_id}"
}

for dataset in hotpotqa musique 2wiki popqa; do
    submit rag "${dataset}"
    submit direct "${dataset}"
done
