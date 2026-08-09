#!/bin/bash
# Submit the locally supported Table 3 matrix after smoke tests pass.
# Usage: bash scripts/launch_table3_evals.sh [smoke|full] [run-tag]

set -euo pipefail

MODE="${1:-smoke}"
RUN_TAG="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"
SBATCH_BIN="${SBATCH_BIN:-/usr/local/slurm/current/bin/sbatch}"
OUTPUT_DIR="${OUTPUT_DIR:-/share/j_sun/rtn27/table3_reproduction}"

case "${MODE}" in
    smoke) NUM_SAMPLES="${NUM_SAMPLES:-2}" ;;
    full) NUM_SAMPLES="${NUM_SAMPLES:-1000}" ;;
    *) echo "Mode must be smoke or full" >&2; exit 2 ;;
esac

# PopQA uses the exact 1,399-row long-tail selection and dedicated launcher in
# scripts/launch_popqa_longtail.sh; including it here would silently run a
# different 1,000-row prefix under the distractor setting.
DATASETS=(hotpotqa musique 2wiki)
BASELINE_METHODS=(direct rag ircot)
BASELINE_MODELS=(qwen3-1.7b qwen3-4b)
KBEVO_MODELS=(kbevo-sft-1.7b kbevo-grpo-1.7b kbevo-sft-4b kbevo-grpo-4b)

submit() {
    local method="$1" model_key="$2" dataset="$3"
    local job_name="t3_${MODE}_${method}_${model_key}_${dataset}"
    local ircot_exports=""
    if [[ "${method}" == "ircot" ]]; then
        : "${IRCOT_RETRIEVER_URL:?Set IRCOT_RETRIEVER_URL before launching IRCoT}"
        ircot_exports=",IRCOT_RETRIEVER_URL=${IRCOT_RETRIEVER_URL},IRCOT_PROMPT_DIR=${IRCOT_PROMPT_DIR:-provenance/ircot/prompts},IRCOT_DISTRACTOR_COUNT=${IRCOT_DISTRACTOR_COUNT:-2}"
    fi
    "${SBATCH_BIN}" \
        --job-name "${job_name:0:120}" \
        --export="ALL,METHOD=${method},MODEL_KEY=${model_key},DATASET=${dataset},NUM_SAMPLES=${NUM_SAMPLES},RUN_TAG=${RUN_TAG},OUTPUT_DIR=${OUTPUT_DIR}${ircot_exports}" \
        scripts/table3_eval_job.slurm
}

for dataset in "${DATASETS[@]}"; do
    for model_key in "${KBEVO_MODELS[@]}"; do
        submit two_phase "${model_key}" "${dataset}"
    done
    for method in "${BASELINE_METHODS[@]}"; do
        if [[ "${method}" == "ircot" && "${dataset}" == "popqa" ]]; then
            continue
        fi
        for model_key in "${BASELINE_MODELS[@]}"; do
            submit "${method}" "${model_key}" "${dataset}"
        done
    done
done
