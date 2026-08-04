#!/bin/bash
#
# Background helper: waits for the NEW-run (icl3hop+tr1024+noth+8gpu)
# step 200 and step 450 merges to finish, then dispatches a 5-benchmark
# eval sweep for whichever finished. Submissions are staggered by 8s to
# avoid the simultaneous-vLLM-startup CUDA-context race that caused 2/5
# failures in the prior batch.

set -uo pipefail

PROJECT_DIR="/home/lz586/icl/Multi-Hop-Reasoning/rebuttal-search-r1"
cd "${PROJECT_DIR}"
LOG="${PROJECT_DIR}/_watch_4b_icl3hop_200_450.log"

dispatched_200=0
dispatched_450=0

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
        # Stagger to avoid simultaneous vLLM CUDA-context startup races
        sleep 8
    done
}

ckpt_ready() {
    [ -f "${PROJECT_DIR}/merged_ckpts/qwen3-4b-icl3hop-tr1024-step$1/config.json" ]
}

echo "[$(date)] Watcher start. Waiting on NEW-run step 200 and 450 merges." > "${LOG}"

for i in $(seq 1 180); do
    if [ "${dispatched_200}" = "0" ] && ckpt_ready 200; then
        dispatch_eval 200
        dispatched_200=1
    fi
    if [ "${dispatched_450}" = "0" ] && ckpt_ready 450; then
        dispatch_eval 450
        dispatched_450=1
    fi
    if [ "${dispatched_200}" = "1" ] && [ "${dispatched_450}" = "1" ]; then
        echo "[$(date)] Both dispatched. Exit." >> "${LOG}"
        exit 0
    fi
    sleep 30
done

echo "[$(date)] Watcher timeout. dispatched_200=${dispatched_200} dispatched_450=${dispatched_450}" >> "${LOG}"
