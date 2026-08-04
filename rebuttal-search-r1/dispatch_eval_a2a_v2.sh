#!/bin/bash
#
# Dispatcher: for each (merged checkpoint, benchmark, split) tuple, submit
# one sbatch_eval_a2a_v2.slurm job. The checkpoint determines which prompt
# variant + enable_thinking the model was trained under.
#
# Pre-reqs (run BEFORE this dispatcher):
#   1) Merge each verl FSDP checkpoint to HF format:
#        python3 -m verl.model_merger merge --backend fsdp \
#            --local_dir /share/j_sun/.../qwen3-1.7b-searchR1-a2a-lr3e6-n8-20260531_1605/global_step_200/actor \
#            --target_dir merged_ckpts/c_icl_step200
#   2) Build per-benchmark eval corpora + indices:
#        for d in "hotpotqa train_val1k" "hotpotqa dev" "musique dev" "2wiki dev" "popqa dev"; do
#            set -- $d
#            python3 prepare_eval_corpus.py --benchmark $1 --split $2 --n_samples 1000
#            python3 build_index_hotpotqa.py \
#                --corpus_path data/eval_corpora/$1_$2/corpus.jsonl \
#                --output_dir  data/eval_corpora/$1_$2
#        done
#
# Usage:
#   bash dispatch_eval_a2a_v2.sh        # submits all
#   DRY_RUN=1 bash dispatch_eval_a2a_v2.sh   # prints what it would submit

set -euo pipefail

PROJECT_DIR="/home/lz586/icl/Multi-Hop-Reasoning/rebuttal-search-r1"
MERGED_ROOT="${PROJECT_DIR}/merged_ckpts"
DRY_RUN="${DRY_RUN:-0}"

# (CKPT_TAG | prompt_variant | enable_thinking)
# CKPT_TAG resolves to ${MERGED_ROOT}/<tag>/ — must exist before submission.
CKPTS=(
    "c_icl_step200|icl3hop|false"        # Run C: ICL + <thinking>, enable_thinking=False
    "d_thinkingtag_step100|thinkingtag|false"  # Run D: <thinking> only, enable_thinking=False
)

# (dataset, split)
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
echo "Dispatch: ${#CKPTS[@]} ckpts × ${#BENCHMARKS[@]} benchmarks = $(( ${#CKPTS[@]} * ${#BENCHMARKS[@]} )) jobs"
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

        JOB_NAME="eval_a2a_${CKPT_TAG}_${DATASET}_${SPLIT}"
        echo ""
        echo "→ ${JOB_NAME}"
        echo "    prompt=${PROMPT_VARIANT}  thinking=${ENABLE_THINKING}"

        if [ "${DRY_RUN}" = "1" ]; then
            echo "    [DRY] would submit"
            continue
        fi

        sbatch \
            -J "${JOB_NAME}" \
            --export=ALL,MERGED_CKPT="${MERGED_CKPT}",DATASET="${DATASET}",SPLIT="${SPLIT}",PROMPT_VARIANT="${PROMPT_VARIANT}",ENABLE_THINKING="${ENABLE_THINKING}" \
            sbatch_eval_a2a_v2.slurm
    done
done

echo ""
echo "=== final queue ==="
squeue -u "${USER}" --format="%.10i %.30j %.10T %.10M %.20R" 2>/dev/null | head -20
