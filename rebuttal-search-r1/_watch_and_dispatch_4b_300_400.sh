#!/bin/bash
#
# Background helper: waits for the step 300 / 400 merges to finish,
# then dispatches a 5-benchmark eval sweep for whichever finished.
# Run via `bash _watch_and_dispatch_4b_300_400.sh &` (or via nohup).

set -uo pipefail

PROJECT_DIR="/home/lz586/icl/Multi-Hop-Reasoning/rebuttal-search-r1"
cd "${PROJECT_DIR}"
LOG="${PROJECT_DIR}/_watch_4b_300_400.log"

dispatched_300=0
dispatched_400=0

dispatch_eval() {
    local STEP="$1"
    local CKPT_TAG="qwen3-4b-searchR1-a2a-lr3e6-step${STEP}"
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

        local JOB_NAME="eval_4b_tr1024_${CKPT_TAG}_${DATASET}_${SPLIT}"
        echo "  -> ${JOB_NAME}" >> "${LOG}"
        sbatch \
            -J "${JOB_NAME}" \
            --export=ALL,MERGED_CKPT="${MERGED}",DATASET="${DATASET}",SPLIT="${SPLIT}",PROMPT_VARIANT=icl3hop,ENABLE_THINKING=false,MAX_TOOL_RESPONSE_LENGTH=1024,TAG=4b_tr1024 \
            sbatch_eval_a2a_v2.slurm >> "${LOG}" 2>&1
        # Stagger submissions: prior batch had 2/5 CUDA-illegal-memory failures
        # when 5 vLLM instances started in lockstep on the same node. A short
        # sleep between submissions lets each vLLM finish its CUDA-context
        # setup before the next one starts.
        sleep 8
    done
}

ckpt_ready() {
    # A merged ckpt is "ready" once config.json exists in the target dir.
    [ -f "${PROJECT_DIR}/merged_ckpts/qwen3-4b-searchR1-a2a-lr3e6-step$1/config.json" ]
}

echo "[$(date)] Watcher start. Waiting on step 300 and 400 merges." > "${LOG}"

# Poll every 30s for up to 90 minutes (180 iterations).
for i in $(seq 1 180); do
    if [ "${dispatched_300}" = "0" ] && ckpt_ready 300; then
        dispatch_eval 300
        dispatched_300=1
    fi
    if [ "${dispatched_400}" = "0" ] && ckpt_ready 400; then
        dispatch_eval 400
        dispatched_400=1
    fi
    if [ "${dispatched_300}" = "1" ] && [ "${dispatched_400}" = "1" ]; then
        echo "[$(date)] Both dispatched. Exit." >> "${LOG}"
        exit 0
    fi
    sleep 30
done

echo "[$(date)] Watcher timeout. dispatched_300=${dispatched_300} dispatched_400=${dispatched_400}" >> "${LOG}"
