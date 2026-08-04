#!/bin/bash
#
# Bootstrap the auto-resume chain for a Search-R1 a2a training run.
#
# This script:
#   1. Submits an initial training sbatch job (with optional resume from a ckpt).
#   2. Submits a watcher sbatch job that depends on the training job
#      (--dependency=afterany), and on training-exit will inspect the status
#      and resubmit if needed.
#
# Usage example (resume Run C 1.7B with ICL data):
#
#   TRAIN_SBATCH=/home/lz586/icl/Multi-Hop-Reasoning/rebuttal-search-r1/sbatch_qwen3_1.7b_apples_to_apples.slurm \
#   LAUNCHER=run_qwen3_1.7b_apples_to_apples.sh \
#   LR=3e-6 \
#   EXPERIMENT_NAME=qwen3-1.7b-searchR1-a2a-lr3e6-n8-20260531_1605 \
#   CKPT_DIR=/share/j_sun/lmlm_multihop/searchr1_checkpoints/search-r1-hotpotqa/qwen3-1.7b-searchR1-a2a-lr3e6-n8-20260531_1605 \
#   RESUME_PATH=/share/j_sun/lmlm_multihop/searchr1_checkpoints/search-r1-hotpotqa/qwen3-1.7b-searchR1-a2a-lr3e6-n8-20260531_1605/global_step_300 \
#   TRAIN_DATA=/home/lz586/icl/Multi-Hop-Reasoning/rebuttal-search-r1/data/train_verl_icl3hop.parquet \
#   VAL_DATA=/home/lz586/icl/Multi-Hop-Reasoning/rebuttal-search-r1/data/test_verl_icl3hop.parquet \
#   NODELIST=aimi-compute-03 \
#   bash /home/lz586/icl/Multi-Hop-Reasoning/scripts/auto_resume_start.sh
#
# Required env:
#   TRAIN_SBATCH    : full path to the training sbatch wrapper
#   LAUNCHER        : launcher .sh (basename or full path; resolved relative to TRAIN_SBATCH dir)
#   LR              : learning rate
#   EXPERIMENT_NAME : experiment name (so resumes write to the same dir)
#   CKPT_DIR        : where checkpoints live (default_local_dir)
#
# Optional env:
#   RESUME_PATH    : initial resume checkpoint (if resuming an existing run)
#   TRAIN_DATA     : training parquet
#   VAL_DATA       : val parquet
#   EXTRA_ARGS     : extra hydra overrides (space-separated)
#   NODELIST       : sbatch --nodelist
#   MAX_RESUMES    : default 10

set -euo pipefail

: "${TRAIN_SBATCH:?TRAIN_SBATCH required (full path)}"
: "${LAUNCHER:?LAUNCHER required}"
: "${LR:?LR required}"
: "${EXPERIMENT_NAME:?EXPERIMENT_NAME required}"
: "${CKPT_DIR:?CKPT_DIR required}"

MAX_RESUMES="${MAX_RESUMES:-10}"
TRAIN_DATA="${TRAIN_DATA:-}"
VAL_DATA="${VAL_DATA:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
NODELIST="${NODELIST:-}"
RESUME_PATH="${RESUME_PATH:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WATCHER_SBATCH="${SCRIPT_DIR}/auto_resume_watcher.slurm"
[ -f "${WATCHER_SBATCH}" ] || { echo "Missing ${WATCHER_SBATCH}"; exit 1; }
[ -f "${TRAIN_SBATCH}" ]   || { echo "Missing ${TRAIN_SBATCH}"; exit 1; }

# Chain log: one file per experiment so multiple chains don't interleave.
CHAIN_LOG_DIR="${SCRIPT_DIR}/watcher_logs"
mkdir -p "${CHAIN_LOG_DIR}"
CHAIN_LOG="${CHAIN_LOG_DIR}/chain_${EXPERIMENT_NAME}.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === auto_resume_start: bootstrapping chain ===" | tee -a "${CHAIN_LOG}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"  | tee -a "${CHAIN_LOG}"
echo "CKPT_DIR=${CKPT_DIR}"                | tee -a "${CHAIN_LOG}"
echo "LR=${LR}"                            | tee -a "${CHAIN_LOG}"
echo "LAUNCHER=${LAUNCHER}"                | tee -a "${CHAIN_LOG}"
echo "RESUME_PATH=${RESUME_PATH:-<none>}"  | tee -a "${CHAIN_LOG}"
echo "TRAIN_DATA=${TRAIN_DATA:-<default>}" | tee -a "${CHAIN_LOG}"
echo "VAL_DATA=${VAL_DATA:-<default>}"     | tee -a "${CHAIN_LOG}"
echo "EXTRA_ARGS=${EXTRA_ARGS:-<none>}"    | tee -a "${CHAIN_LOG}"
echo "NODELIST=${NODELIST:-<any>}"         | tee -a "${CHAIN_LOG}"
echo "MAX_RESUMES=${MAX_RESUMES}"          | tee -a "${CHAIN_LOG}"

# ----- 1. Submit initial training -----
EXPORT_LIST="ALL,LR=${LR},LAUNCHER=${LAUNCHER},EXPERIMENT_NAME=${EXPERIMENT_NAME}"
[ -n "${RESUME_PATH}" ] && EXPORT_LIST="${EXPORT_LIST},RESUME_PATH=${RESUME_PATH}"
[ -n "${TRAIN_DATA}" ]  && EXPORT_LIST="${EXPORT_LIST},TRAIN_DATA=${TRAIN_DATA}"
[ -n "${VAL_DATA}" ]    && EXPORT_LIST="${EXPORT_LIST},VAL_DATA=${VAL_DATA}"
[ -n "${EXTRA_ARGS}" ]  && EXPORT_LIST="${EXPORT_LIST},EXTRA_ARGS=${EXTRA_ARGS}"

TRAIN_SBATCH_ARGS=(--parsable --export="${EXPORT_LIST}")
[ -n "${NODELIST}" ]            && TRAIN_SBATCH_ARGS+=(--nodelist="${NODELIST}")
[ -n "${SBATCH_GRES:-}" ]       && TRAIN_SBATCH_ARGS+=(--gres="${SBATCH_GRES}")

TRAIN_JOBID=$(sbatch "${TRAIN_SBATCH_ARGS[@]}" "${TRAIN_SBATCH}")
if [ -z "${TRAIN_JOBID}" ]; then
    echo "[$(date)] sbatch returned empty jobid for training. Aborting." | tee -a "${CHAIN_LOG}"
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Submitted initial training: ${TRAIN_JOBID}" | tee -a "${CHAIN_LOG}"

# ----- 2. Submit first watcher (depends on training) -----
WATCHER_EXPORT="ALL,WATCH_JOBID=${TRAIN_JOBID},CKPT_DIR=${CKPT_DIR},TRAIN_SBATCH=${TRAIN_SBATCH},LAUNCHER=${LAUNCHER},EXPERIMENT_NAME=${EXPERIMENT_NAME},LR=${LR},CHAIN_LOG=${CHAIN_LOG},MAX_RESUMES=${MAX_RESUMES},RESUME_COUNT=0,PREV_LAST_STEP=-1"
[ -n "${TRAIN_DATA}" ] && WATCHER_EXPORT="${WATCHER_EXPORT},TRAIN_DATA=${TRAIN_DATA}"
[ -n "${VAL_DATA}" ]   && WATCHER_EXPORT="${WATCHER_EXPORT},VAL_DATA=${VAL_DATA}"
[ -n "${EXTRA_ARGS}" ] && WATCHER_EXPORT="${WATCHER_EXPORT},EXTRA_ARGS=${EXTRA_ARGS}"
[ -n "${NODELIST}" ]   && WATCHER_EXPORT="${WATCHER_EXPORT},NODELIST=${NODELIST}"

WATCHER_JOBID=$(sbatch --parsable \
    --dependency=afterany:"${TRAIN_JOBID}" \
    --export="${WATCHER_EXPORT}" \
    "${WATCHER_SBATCH}")
if [ -z "${WATCHER_JOBID}" ]; then
    echo "[$(date)] WARNING: could not submit first watcher. Training will run unattended." | tee -a "${CHAIN_LOG}"
    exit 1
fi
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Submitted first watcher: ${WATCHER_JOBID} (--dependency=afterany:${TRAIN_JOBID})" | tee -a "${CHAIN_LOG}"
echo
echo "Chain log: ${CHAIN_LOG}"
echo "  tail -f ${CHAIN_LOG}"
echo "Initial training job: ${TRAIN_JOBID}"
echo "First watcher job:    ${WATCHER_JOBID}"
