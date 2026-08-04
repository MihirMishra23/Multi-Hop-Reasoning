#!/bin/bash
#
# Dispatcher for the Qwen3-4B Search-R1 a2a lr3e6 step=500 ckpt
# (qwen3-4b-searchR1-a2a-lr3e6-n8-20260530_1023/global_step_500).
#
# Same 5-benchmark per-benchmark-corpus suite as the 1.7B tr1024 sweep,
# with MAX_TOOL_RESPONSE_LENGTH=1024 to match the 4B training-time
# truncation (and to match what the 1.7B sweep used, for direct comparison).
#
# Uses the patched eval_search_r1.py extract_answer that also recognizes
# answers emitted as <tool_call>{"name":"answer",...}</tool_call> — the
# 4B model drifts to this format heavily, which is why the pre-patch
# eval produced suspiciously low EM (e.g. 0.145 on 2wiki).
#
# Usage:
#   bash dispatch_eval_a2a_4b_tr1024.sh          # submits all 5
#   DRY_RUN=1 bash dispatch_eval_a2a_4b_tr1024.sh

set -euo pipefail

PROJECT_DIR="/home/lz586/icl/Multi-Hop-Reasoning/rebuttal-search-r1"
MERGED_ROOT="${PROJECT_DIR}/merged_ckpts"
DRY_RUN="${DRY_RUN:-0}"

CKPTS=(
    "qwen3-4b-searchR1-a2a-lr3e6-step500|icl3hop|false"
)

BENCHMARKS=(
    "hotpotqa train_val1k"
    "hotpotqa dev"
    "musique dev"
    "2wiki dev"
    "popqa dev"
)

cd "${PROJECT_DIR}"
mkdir -p eval_results

echo "=========================================="
echo "4B tr1024 dispatch: ${#CKPTS[@]} ckpt(s) x ${#BENCHMARKS[@]} benchmarks = $(( ${#CKPTS[@]} * ${#BENCHMARKS[@]} )) jobs"
echo "MAX_TOOL_RESPONSE_LENGTH=1024 (matches training)"
echo "DRY_RUN=${DRY_RUN}"
echo "=========================================="

for CKPT_LINE in "${CKPTS[@]}"; do
    IFS='|' read -r CKPT_TAG PROMPT_VARIANT ENABLE_THINKING <<< "${CKPT_LINE}"
    MERGED_CKPT="${MERGED_ROOT}/${CKPT_TAG}"

    if [ ! -d "${MERGED_CKPT}" ]; then
        echo "  SKIP ${CKPT_TAG}: ${MERGED_CKPT} not found (merge ckpt first)"
        continue
    fi

    for BM in "${BENCHMARKS[@]}"; do
        set -- ${BM}
        DATASET="$1"
        SPLIT="$2"

        CORPUS_PATH="${PROJECT_DIR}/data/eval_corpora/${DATASET}_${SPLIT}/corpus.jsonl"
        INDEX_PATH="${PROJECT_DIR}/data/eval_corpora/${DATASET}_${SPLIT}/e5_Flat.index"
        if [ ! -f "${CORPUS_PATH}" ] || [ ! -f "${INDEX_PATH}" ]; then
            echo "  SKIP ${CKPT_TAG} x ${DATASET}/${SPLIT}: missing corpus or index (${CORPUS_PATH})"
            continue
        fi

        JOB_NAME="eval_4b_tr1024_${CKPT_TAG}_${DATASET}_${SPLIT}"
        echo ""
        echo "-> ${JOB_NAME}"
        echo "    prompt=${PROMPT_VARIANT}  thinking=${ENABLE_THINKING}  tr=1024"

        if [ "${DRY_RUN}" = "1" ]; then
            echo "    [DRY] would submit"
            continue
        fi

        sbatch \
            -J "${JOB_NAME}" \
            --export=ALL,MERGED_CKPT="${MERGED_CKPT}",DATASET="${DATASET}",SPLIT="${SPLIT}",PROMPT_VARIANT="${PROMPT_VARIANT}",ENABLE_THINKING="${ENABLE_THINKING}",MAX_TOOL_RESPONSE_LENGTH=1024,TAG=4b_tr1024 \
            sbatch_eval_a2a_v2.slurm
    done
done

echo ""
echo "=== final queue ==="
squeue -u "${USER}" --format="%.10i %.40j %.10T %.10M %.20R" 2>/dev/null | head -30
