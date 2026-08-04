#!/bin/bash
#
# Wait for NEW-run 4B icl3hop step 500 merge to finish, then dispatch
# the 5-benchmark eval sweep (with 8s inter-job stagger).

set -uo pipefail

PROJECT_DIR="/home/lz586/icl/Multi-Hop-Reasoning/rebuttal-search-r1"
cd "${PROJECT_DIR}"
LOG="${PROJECT_DIR}/_watch_4b_icl3hop_500.log"

dispatched=0

dispatch_eval() {
    local STEP="$1"
    local CKPT_TAG="qwen3-4b-icl3hop-tr1024-step${STEP}"
    local MERGED="${PROJECT_DIR}/merged_ckpts/${CKPT_TAG}"
    echo "[$(date)] Dispatching eval for step ${STEP}" >> "${LOG}"

    for BM in "hotpotqa train_val1k" "hotpotqa dev" "musique dev" "2wiki dev" "popqa dev"; do
        set -- ${BM}
        local DATASET="$1"
        local SPLIT="$2"
        local CORPUS="${PROJECT_DIR}/data/eval_corpora/${DATASET}_${SPLIT}/corpus.jsonl"
        local INDEX="${PROJECT_DIR}/data/eval_corpora/${DATASET}_${SPLIT}/e5_Flat.index"
        if [ ! -f "${CORPUS}" ] || [ ! -f "${INDEX}" ]; then
            echo "  SKIP ${CKPT_TAG} x ${DATASET}/${SPLIT}: missing corpus or index" >> "${LOG}"
            continue
        fi

        local JOB_NAME="eval_4b_icl3hop_${CKPT_TAG}_${DATASET}_${SPLIT}"
        echo "  -> ${JOB_NAME}" >> "${LOG}"
        sbatch \
            -J "${JOB_NAME}" \
            --export=ALL,MERGED_CKPT="${MERGED}",DATASET="${DATASET}",SPLIT="${SPLIT}",PROMPT_VARIANT=icl3hop,ENABLE_THINKING=false,MAX_TOOL_RESPONSE_LENGTH=1024,TAG=4b_icl3hop \
            sbatch_eval_a2a_v2.slurm >> "${LOG}" 2>&1
        sleep 8
    done
}

ckpt_ready() {
    [ -f "${PROJECT_DIR}/merged_ckpts/qwen3-4b-icl3hop-tr1024-step$1/config.json" ]
}

echo "[$(date)] Watcher start. Waiting on NEW-run step 500 merge." > "${LOG}"

for i in $(seq 1 60); do
    if [ "${dispatched}" = "0" ] && ckpt_ready 500; then
        dispatch_eval 500
        dispatched=1
        echo "[$(date)] Dispatched. Exit." >> "${LOG}"
        exit 0
    fi
    sleep 30
done

echo "[$(date)] Watcher timeout. dispatched=${dispatched}" >> "${LOG}"
