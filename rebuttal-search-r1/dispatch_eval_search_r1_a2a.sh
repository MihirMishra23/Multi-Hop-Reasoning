#!/bin/bash
#
# Apples-to-apples Search-R1 eval dispatcher.
#
# What this does
# --------------
# For every merged Search-R1 checkpoint in MODEL_PATH_LST, submits ONE sbatch
# job per (dataset, split) pair via sbatch_eval_search_r1.slurm. Each sub-job:
#   - Runs eval_search_r1.py against the same questions as the LMLM eval
#     (matching DATASET_SPLIT_MAP inside eval_search_r1.py).
#   - Samples at T=1.0, top_p=0.95, top_k=4 (the Search-R1 a2a training recipe;
#     also matches LMLM eval's --use-train-params).
#   - Uses max_response_length=2048 and max_assistant_turns=5 (training values).
#   - Calls the local /retrieve server with topk=3 (verl default; same as
#     training tool config).
#
# Apples-to-apples vs LMLM
# ------------------------
#   - Same questions: get_dataset(...) loader is shared.
#   - Same metric: EM uses src/eval/metrics.exact_match_score; F1 uses
#     SearchR1's hotpotqa_f1.py (the training reward, max over golds).
#   - Same num_samples: 1000.
#   - Same sampling distribution: T=1.0, top_p=0.95 in both.
#
# Submission granularity
# ----------------------
# Mirrors LMLM eval_lmlm.slurm's `per_ckpt_benchmark`: one job per
# (checkpoint, benchmark). hotpotqa runs both train_val1k and dev as two
# separate jobs (no concept of grouping splits inside the eval script).
#
# Usage
# -----
#   bash dispatch_eval_search_r1_a2a.sh             # submit
#   SUBMIT_MODE=dryrun bash dispatch_eval_search_r1_a2a.sh   # print only
#
# Env overrides (rare):
#   NODE_LIST=aimi-compute-02 NUM_SAMPLES=100 bash dispatch_eval_search_r1_a2a.sh

set -euo pipefail

PROJECT_DIR="/home/lz586/icl/Multi-Hop-Reasoning/rebuttal-search-r1"
SLURM_SCRIPT="${PROJECT_DIR}/sbatch_eval_search_r1.slurm"

# ── Checkpoints to evaluate ─────────────────────────────────────────────────
# Add merged HF dirs here. Each must contain config.json + model.safetensors.
MODEL_PATH_LST=(
    "${PROJECT_DIR}/merged_ckpts/qwen3-1.7b-icl3hop-gen4096-step250"
    "${PROJECT_DIR}/merged_ckpts/qwen3-1.7b-icl3hop-gen4096-step500"
)

# ── Benchmarks: match the LMLM eval matrix ─────────────────────────────────
# (DATASET, SPLIT) pairs — eval_search_r1.py:DATASET_SPLIT_MAP must support each.
BENCHMARKS=(
    "hotpotqa:dev"
    "hotpotqa:train_val1k"
    "musique:dev"
    "2wiki:dev"
    "trivia_qa:dev"
    "popqa:dev"
)

# ── Defaults (all overridable via env) ──────────────────────────────────────
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
SUBMIT_MODE="${SUBMIT_MODE:-launch}"  # launch | dryrun
NODE_LIST="${NODE_LIST:-}"             # e.g. aimi-compute-02 to pin
TAG_SUFFIX="${TAG_SUFFIX:-a2a}"        # filename suffix to flag a2a results
PARTITION="${PARTITION:-aimi}"         # aimi (B200) is fast; override to kilian/default_partition for A6000
GRES="${GRES:-gpu:nvidia_b200:1}"
SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-}"  # e.g. afterok:967929 to wait for a merge job

# ── Submit ──────────────────────────────────────────────────────────────────
for MERGED_CKPT in "${MODEL_PATH_LST[@]}"; do
    if [ ! -f "${MERGED_CKPT}/config.json" ]; then
        if [ -n "${SBATCH_DEPENDENCY}" ]; then
            echo "WARN (no config.json yet, but SBATCH_DEPENDENCY set, will queue): ${MERGED_CKPT}" >&2
        else
            echo "SKIP (no config.json): ${MERGED_CKPT}" >&2
            continue
        fi
    fi
    for B in "${BENCHMARKS[@]}"; do
        DATASET="${B%%:*}"
        SPLIT="${B##*:}"
        CKPT_TAG="$(basename "${MERGED_CKPT}")"
        JOB_NAME="evalA2A_${CKPT_TAG}_${DATASET}_${SPLIT}"

        SBATCH_ARGS=(
            -J "${JOB_NAME}"
            --partition="${PARTITION}"
            --gres="${GRES}"
        )
        if [ -n "${NODE_LIST}" ]; then
            SBATCH_ARGS+=(--nodelist="${NODE_LIST}")
        fi
        if [ -n "${SBATCH_DEPENDENCY}" ]; then
            SBATCH_ARGS+=(--dependency="${SBATCH_DEPENDENCY}")
        fi

        # Per-job env passed to sbatch_eval_search_r1.slurm
        ENV_KV=(
            "MERGED_CKPT=${MERGED_CKPT}"
            "DATASET=${DATASET}"
            "SPLIT=${SPLIT}"
            "NUM_SAMPLES=${NUM_SAMPLES}"
            "TAG=${TAG_SUFFIX}"
        )
        # Pass through optional length overrides (e.g. for gen=4096 ckpts).
        [ -n "${MAX_RESPONSE_LENGTH:-}" ] && ENV_KV+=("MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}")
        [ -n "${MAX_MODEL_LEN:-}" ]       && ENV_KV+=("MAX_MODEL_LEN=${MAX_MODEL_LEN}")
        [ -n "${MAX_TURNS:-}" ]           && ENV_KV+=("MAX_TURNS=${MAX_TURNS}")

        if [ "${SUBMIT_MODE}" = "dryrun" ]; then
            echo "[dryrun] sbatch ${SBATCH_ARGS[*]} (env: ${ENV_KV[*]}) ${SLURM_SCRIPT}"
        else
            echo "Submitting: ${JOB_NAME}"
            env "${ENV_KV[@]}" sbatch "${SBATCH_ARGS[@]}" "${SLURM_SCRIPT}"
        fi
    done
done

echo "Dispatcher done."
