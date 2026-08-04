#!/bin/bash
#
# Wiki18 full-corpus dispatcher: same 2 ckpts × 5 benchmarks as
# dispatch_eval_a2a_v2.sh, but the retrieval datastore is the FULL 21M-passage
# wiki18 corpus instead of a per-benchmark 1k-12k passage slice.
#
# Why: the per-benchmark eval gave very high EM on hotpotqa/musique (because
# the gold passages always exist in the small datastore — leakage). We want
# the realistic perf gap on a full wiki backend.
#
# Datastore: /share/nikola/wiki18/{wiki18_100w.jsonl, e5_flat_inner.index}
# - 21,015,324 passages, dim 768, IndexFlatIP (e5-base-v2)
# - 64.5 GB index, 14 GB corpus jsonl
# - Verified equivalent to /share/aimi/phantom-agent/data/searchr1/e5_Flat.index
#
# Usage:
#   bash dispatch_eval_a2a_wiki18.sh            # submits all 10
#   SMOKE=1 bash dispatch_eval_a2a_wiki18.sh    # only the first ckpt × first benchmark
#   DRY_RUN=1 bash dispatch_eval_a2a_wiki18.sh  # prints what it would submit

set -euo pipefail

PROJECT_DIR="/home/lz586/icl/Multi-Hop-Reasoning/rebuttal-search-r1"
MERGED_ROOT="${PROJECT_DIR}/merged_ckpts"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"

# Full wiki18 datastore (world-readable, no copy needed).
WIKI18_CORPUS="/share/nikola/wiki18/wiki18_100w.jsonl"
WIKI18_INDEX="/share/nikola/wiki18/e5_flat_inner.index"

for f in "${WIKI18_CORPUS}" "${WIKI18_INDEX}"; do
    [ -f "${f}" ] || { echo "MISSING: ${f}"; exit 1; }
done

# (CKPT_TAG | prompt_variant | enable_thinking)
CKPTS=(
    "c_icl_step200|icl3hop|false"
    "d_thinkingtag_step100|thinkingtag|false"
)

BENCHMARKS=(
    "hotpotqa train_val1k"
    "hotpotqa dev"
    "musique dev"
    "2wiki dev"
    "popqa dev"
)

if [ "${SMOKE}" = "1" ]; then
    echo "SMOKE=1 — only submitting first ckpt × first benchmark"
    CKPTS=("${CKPTS[0]}")
    BENCHMARKS=("${BENCHMARKS[0]}")
fi

cd "${PROJECT_DIR}"
mkdir -p eval_results

echo "=========================================="
echo "Wiki18 dispatch: ${#CKPTS[@]} ckpts × ${#BENCHMARKS[@]} benchmarks = $(( ${#CKPTS[@]} * ${#BENCHMARKS[@]} )) jobs"
echo "DRY_RUN=${DRY_RUN}   SMOKE=${SMOKE}"
echo "CORPUS:  ${WIKI18_CORPUS}"
echo "INDEX:   ${WIKI18_INDEX}"
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

        JOB_NAME="eval_wiki18_${CKPT_TAG}_${DATASET}_${SPLIT}"
        echo ""
        echo "→ ${JOB_NAME}"
        echo "    prompt=${PROMPT_VARIANT}  thinking=${ENABLE_THINKING}"

        if [ "${DRY_RUN}" = "1" ]; then
            echo "    [DRY] would submit"
            continue
        fi

        # --mem=256G: 64GB index + retrieval server + buffers
        # -t 6:00:00: extra wall time (retrieval is slower on full index)
        # RETRIEVAL_HEALTH_TIMEOUT=600: 64GB mmap from NFS takes minutes
        # TAG=wiki18: so output filenames are distinguishable
        sbatch \
            -J "${JOB_NAME}" \
            --mem=256G \
            -t 6:00:00 \
            --export=ALL,MERGED_CKPT="${MERGED_CKPT}",DATASET="${DATASET}",SPLIT="${SPLIT}",PROMPT_VARIANT="${PROMPT_VARIANT}",ENABLE_THINKING="${ENABLE_THINKING}",CORPUS_PATH="${WIKI18_CORPUS}",INDEX_PATH="${WIKI18_INDEX}",RETRIEVAL_HEALTH_TIMEOUT=600,TAG=wiki18 \
            sbatch_eval_a2a_v2.slurm
    done
done

echo ""
echo "=== final queue ==="
squeue -u "${USER}" --format="%.10i %.30j %.10T %.10M %.20R" 2>/dev/null | head -30
