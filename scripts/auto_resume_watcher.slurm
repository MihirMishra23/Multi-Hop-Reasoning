#!/bin/bash
#
# Auto-resume watcher for Search-R1 a2a training.
#
# Submitted with --dependency=afterany:<TRAIN_JOBID> by either
# auto_resume_start.sh (chain bootstrap) or by a previous watcher iteration.
# Runs after the watched training job exits (any status). Reads sacct, and:
#   - if COMPLETED  -> exit (chain done)
#   - if FAILED/OOM/TIMEOUT/NODE_FAIL/PREEMPTED -> find latest checkpoint,
#     resubmit training + chain next watcher
#   - if CANCELLED  -> exit (operator killed it deliberately)
#   - if RESUME_COUNT >= MAX_RESUMES -> exit (safety cap)
#   - if latest ckpt step did not advance since last resume -> exit
#     (deterministic crash before any checkpoint; don't loop)
#
# Required env (passed via --export):
#   WATCH_JOBID     : the training job we just watched (the one that exited)
#   CKPT_DIR        : where to find global_step_N (the training job's default_local_dir)
#   TRAIN_SBATCH    : path to the training sbatch wrapper (.slurm)
#   LAUNCHER        : launcher .sh script (e.g. run_qwen3_1.7b_apples_to_apples.sh)
#   EXPERIMENT_NAME : experiment name (so we write into the same ckpt dir)
#   LR              : learning rate
#   CHAIN_LOG       : log file for the whole chain (appended)
#
# Optional env:
#   TRAIN_DATA     : override training parquet (default: launcher's default)
#   VAL_DATA       : override val parquet
#   EXTRA_ARGS     : extra hydra overrides (space-separated)
#   NODELIST       : sbatch --nodelist for the next training job
#   MAX_RESUMES    : default 10
#   RESUME_COUNT   : default 0 (incremented each iteration)
#   PREV_LAST_STEP : last seen checkpoint step (default -1; -1 means no prior)

#SBATCH -J a2a_watcher
#SBATCH -o /home/lz586/icl/slurm_output/searchr1_a2a/%x_%j.out
#SBATCH -e /home/lz586/icl/slurm_output/searchr1_a2a/%x_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH -t 00:10:00
#SBATCH --partition=aimi

set -uo pipefail

: "${WATCH_JOBID:?WATCH_JOBID required}"
: "${CKPT_DIR:?CKPT_DIR required}"
: "${TRAIN_SBATCH:?TRAIN_SBATCH required}"
: "${LAUNCHER:?LAUNCHER required}"
: "${EXPERIMENT_NAME:?EXPERIMENT_NAME required}"
: "${LR:?LR required}"
: "${CHAIN_LOG:?CHAIN_LOG required}"

MAX_RESUMES="${MAX_RESUMES:-10}"
RESUME_COUNT="${RESUME_COUNT:-0}"
PREV_LAST_STEP="${PREV_LAST_STEP:--1}"
TRAIN_DATA="${TRAIN_DATA:-}"
VAL_DATA="${VAL_DATA:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
NODELIST="${NODELIST:-}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [watcher $SLURM_JOB_ID] $*" | tee -a "${CHAIN_LOG}"; }

log "=== Watcher fired for WATCH_JOBID=${WATCH_JOBID} (resume_count=${RESUME_COUNT}/${MAX_RESUMES}) ==="

# ----- 1. Inspect the watched job's exit status -----
# sacct may take a few seconds to flush; retry briefly.
STATE=""
for attempt in $(seq 1 6); do
    STATE=$(sacct -j "${WATCH_JOBID}" --format=State -P -n 2>/dev/null | head -1 | awk '{print $1}')
    if [ -n "${STATE}" ]; then break; fi
    sleep 5
done
log "Watched job state: ${STATE:-UNKNOWN}"

case "${STATE}" in
    COMPLETED)
        log "Training completed cleanly. Chain done."
        exit 0
        ;;
    CANCELLED|CANCELLED+)
        log "Training was cancelled (likely by operator). Chain stopped."
        exit 0
        ;;
    FAILED|OUT_OF_MEMORY|TIMEOUT|NODE_FAIL|PREEMPTED|"")
        log "Training did not complete cleanly (state=${STATE:-UNKNOWN}). Attempting resume."
        ;;
    *)
        log "Unexpected state '${STATE}'. Attempting resume anyway."
        ;;
esac

# ----- 2. Safety: cap on resumes -----
if [ "${RESUME_COUNT}" -ge "${MAX_RESUMES}" ]; then
    log "RESUME_COUNT=${RESUME_COUNT} >= MAX_RESUMES=${MAX_RESUMES}. Stopping chain."
    exit 0
fi

# ----- 3. Find latest checkpoint -----
if [ ! -d "${CKPT_DIR}" ]; then
    log "CKPT_DIR=${CKPT_DIR} does not exist. Cannot resume. Stopping."
    exit 0
fi
LATEST_CKPT_PATH=$(ls -1d "${CKPT_DIR}"/global_step_* 2>/dev/null \
    | awk -F'global_step_' '{print $2, $0}' | sort -n | tail -1 | awk '{print $2}')
if [ -z "${LATEST_CKPT_PATH:-}" ]; then
    log "No global_step_* found in ${CKPT_DIR}. Cannot resume. Stopping."
    exit 0
fi
LATEST_STEP=$(basename "${LATEST_CKPT_PATH}" | sed 's/global_step_//')
log "Latest checkpoint: ${LATEST_CKPT_PATH} (step ${LATEST_STEP})"

# ----- 4. Loop guard: ckpt must advance between resumes -----
if [ "${PREV_LAST_STEP}" != "-1" ] && [ "${LATEST_STEP}" -le "${PREV_LAST_STEP}" ]; then
    log "Checkpoint did not advance (prev=${PREV_LAST_STEP}, now=${LATEST_STEP}). Looks like a deterministic crash before save_freq=50 hits. Stopping to avoid infinite loop."
    exit 0
fi

# ----- 5. Resubmit training -----
NEW_RESUME_COUNT=$((RESUME_COUNT + 1))
log "Resubmitting training (resume #${NEW_RESUME_COUNT}) from ${LATEST_CKPT_PATH}"

EXPORT_LIST="ALL,LR=${LR},LAUNCHER=${LAUNCHER},RESUME_PATH=${LATEST_CKPT_PATH},EXPERIMENT_NAME=${EXPERIMENT_NAME}"
[ -n "${TRAIN_DATA}" ] && EXPORT_LIST="${EXPORT_LIST},TRAIN_DATA=${TRAIN_DATA}"
[ -n "${VAL_DATA}" ]   && EXPORT_LIST="${EXPORT_LIST},VAL_DATA=${VAL_DATA}"
[ -n "${EXTRA_ARGS}" ] && EXPORT_LIST="${EXPORT_LIST},EXTRA_ARGS=${EXTRA_ARGS}"

SBATCH_ARGS=(--parsable --export="${EXPORT_LIST}")
[ -n "${NODELIST}" ]      && SBATCH_ARGS+=(--nodelist="${NODELIST}")
[ -n "${SBATCH_GRES:-}" ] && SBATCH_ARGS+=(--gres="${SBATCH_GRES}")

NEW_TRAIN_JOBID=$(sbatch "${SBATCH_ARGS[@]}" "${TRAIN_SBATCH}")
if [ -z "${NEW_TRAIN_JOBID}" ]; then
    log "sbatch returned empty jobid for training. Stopping chain."
    exit 1
fi
log "Submitted training job: ${NEW_TRAIN_JOBID}"

# ----- 6. Chain next watcher -----
WATCHER_SBATCH="$(readlink -f "${BASH_SOURCE[0]}")"
WATCHER_EXPORT="ALL,WATCH_JOBID=${NEW_TRAIN_JOBID},CKPT_DIR=${CKPT_DIR},TRAIN_SBATCH=${TRAIN_SBATCH},LAUNCHER=${LAUNCHER},EXPERIMENT_NAME=${EXPERIMENT_NAME},LR=${LR},CHAIN_LOG=${CHAIN_LOG},MAX_RESUMES=${MAX_RESUMES},RESUME_COUNT=${NEW_RESUME_COUNT},PREV_LAST_STEP=${LATEST_STEP}"
[ -n "${TRAIN_DATA}" ] && WATCHER_EXPORT="${WATCHER_EXPORT},TRAIN_DATA=${TRAIN_DATA}"
[ -n "${VAL_DATA}" ]   && WATCHER_EXPORT="${WATCHER_EXPORT},VAL_DATA=${VAL_DATA}"
[ -n "${EXTRA_ARGS}" ] && WATCHER_EXPORT="${WATCHER_EXPORT},EXTRA_ARGS=${EXTRA_ARGS}"
[ -n "${NODELIST}" ]   && WATCHER_EXPORT="${WATCHER_EXPORT},NODELIST=${NODELIST}"

NEW_WATCHER_JOBID=$(sbatch --parsable \
    --dependency=afterany:"${NEW_TRAIN_JOBID}" \
    --export="${WATCHER_EXPORT}" \
    "${WATCHER_SBATCH}")
if [ -z "${NEW_WATCHER_JOBID}" ]; then
    log "WARNING: could not chain next watcher. Training will run unattended."
else
    log "Chained next watcher: ${NEW_WATCHER_JOBID} (--dependency=afterany:${NEW_TRAIN_JOBID})"
fi

log "Watcher done."
