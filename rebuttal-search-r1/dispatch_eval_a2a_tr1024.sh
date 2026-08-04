#!/bin/bash
#
# Dispatcher for the icl3hop-tr1024-noth run
# (qwen3-1.7b-searchR1-a2a-lr3e6-n8-icl3hop-tr1024-noth-20260601_0416).
#
# vs the original v2 dispatcher: same 5-benchmark per-benchmark-corpus suite,
# but MAX_TOOL_RESPONSE_LENGTH=1024 to match the training-time truncation
# (the canonical Run C used 512; this run trained with 1024, so eval must
# match or we recreate the training-eval distribution mismatch bug).
#
# Steps evaluated: 200 (head-to-head with c_icl_step200) and 500 (final).
#
# Usage:
#   bash dispatch_eval_a2a_tr1024.sh          # submits all 10
#   DRY_RUN=1 bash dispatch_eval_a2a_tr1024.sh

set -euo pipefail

PROJECT_DIR="/home/lz586/icl/Multi-Hop-Reasoning/rebuttal-search-r1"
MERGED_ROOT="${PROJECT_DIR}/merged_ckpts"
DRY_RUN="${DRY_RUN:-0}"

# (CKPT_TAG | prompt_variant | enable_thinking)
CKPTS=(
    "icl3hop_tr1024_step200|icl3hop|false"
    "icl3hop_tr1024_step500|icl3hop|false"
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
echo "tr1024 dispatch: ${#CKPTS[@]} ckpts × ${#BENCHMARKS[@]} benchmarks = $(( ${#CKPTS[@]} * ${#BENCHMARKS[@]} )) jobs"
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
            echo "  SKIP ${CKPT_TAG} × ${DATASET}/${SPLIT}: missing corpus or index ($CORPUS_PATH)"
            continue
        fi

        JOB_NAME="eval_tr1024_${CKPT_TAG}_${DATASET}_${SPLIT}"
        echo ""
        echo "→ ${JOB_NAME}"
        echo "    prompt=${PROMPT_VARIANT}  thinking=${ENABLE_THINKING}  tr=1024"

        if [ "${DRY_RUN}" = "1" ]; then
            echo "    [DRY] would submit"
            continue
        fi

        sbatch \
            -J "${JOB_NAME}" \
            --export=ALL,MERGED_CKPT="${MERGED_CKPT}",DATASET="${DATASET}",SPLIT="${SPLIT}",PROMPT_VARIANT="${PROMPT_VARIANT}",ENABLE_THINKING="${ENABLE_THINKING}",MAX_TOOL_RESPONSE_LENGTH=1024,TAG=tr1024 \
            sbatch_eval_a2a_v2.slurm
    done
done

echo ""
echo "=== final queue ==="
squeue -u "${USER}" --format="%.10i %.40j %.10T %.10M %.20R" 2>/dev/null | head -30
