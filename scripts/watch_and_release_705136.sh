#!/usr/bin/env bash
# Releases SLURM interactive job 705136 once the wandb run output goes idle.
#
# Submit with:
#   sbatch /home/lz586/icl/Multi-Hop-Reasoning/scripts/watch_and_release_705136.sh
#
# SLURM Directives
#SBATCH -J watch_release_705136
#SBATCH -o /home/lz586/icl/slurm_output/lmlm_multihop/watch_release_705136_%j.out
#SBATCH -e /home/lz586/icl/slurm_output/lmlm_multihop/watch_release_705136_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH -t 12:00:00
#SBATCH --partition=aimi,kilian,jjs533,default_partition

set -u

TARGET_JOBID=705136
LOG=/home/lz586/icl/Multi-Hop-Reasoning/wandb/run-20260524_164225-0tzfsqyj/files/output.log
IDLE_SECS=600       # 10 min of no writes => training assumed finished
POLL_SECS=60

echo "[$(date)] watcher started (slurm jobid=${SLURM_JOB_ID:-N/A}) on $(hostname)"
echo "          target_jobid=$TARGET_JOBID, log=$LOG, idle_threshold=${IDLE_SECS}s"

while true; do
  if [[ ! -f "$LOG" ]]; then
    echo "[$(date)] log missing, aborting watcher"
    exit 1
  fi
  last=$(stat -c %Y "$LOG")
  now=$(date +%s)
  delta=$(( now - last ))
  if (( delta > IDLE_SECS )); then
    echo "[$(date)] log idle ${delta}s > ${IDLE_SECS}s, scancel $TARGET_JOBID"
    scancel "$TARGET_JOBID"
    exit 0
  fi
  sleep "$POLL_SECS"
done
