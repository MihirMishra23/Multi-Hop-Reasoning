#!/bin/bash
#
# Same hardening as the sbatch wrapper, but for tmux launches on aimi-01.
# Snapshots verl_config to node-local /scratch so NFS edits/deletes can't kill
# a running training job (the 2026-05-27 01:03 crash root cause).
#
# Run from inside a tmux session, NOT directly — the trap cleanup only fires
# on normal exit, and the retrieval server is assumed to already be up at :8000.

set -euo pipefail

ENV_DIR="/scratch/lz586/envs/searchr1-sgl"
PROJECT_DIR="/home/lz586/icl/Multi-Hop-Reasoning/rebuttal-search-r1"
RUN_ID="tmux_$(date +%Y%m%d_%H%M%S)_$$"
LOCAL_RUN_DIR="/scratch/lz586/run_${RUN_ID}"
LOCAL_CONFIG_PATH="${LOCAL_RUN_DIR}/verl_config"
TRAIN_LOG="${PROJECT_DIR}/logs/train_a2a_4b_${RUN_ID}.log"

export PATH="${ENV_DIR}/bin:${PATH}"
cd "${PROJECT_DIR}"

# Snapshot verl_config to node-local /scratch
echo "[$(date)] Snapshotting verl_config -> ${LOCAL_CONFIG_PATH}"
mkdir -p "${LOCAL_RUN_DIR}"
cp -r "${PROJECT_DIR}/verl_config" "${LOCAL_CONFIG_PATH}"
ls "${LOCAL_CONFIG_PATH}/tool_config/" >/dev/null || { echo "snapshot failed"; exit 1; }

cleanup() {
    echo "[$(date)] Cleaning up local run dir ${LOCAL_RUN_DIR}"
    rm -rf "${LOCAL_RUN_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

# Preflight: retrieval server should already be up
curl -sf -m 3 http://127.0.0.1:8000/health > /dev/null \
    || { echo "retrieval not responding at :8000 — start launch_retrieval_verl.sh first"; exit 1; }

echo "[$(date)] Launching 4B training -> ${TRAIN_LOG}"
CONFIG_PATH="${LOCAL_CONFIG_PATH}" \
TOOL_CONFIG="${LOCAL_CONFIG_PATH}/tool_config/search_tool_config.yaml" \
    bash run_qwen3_4b_apples_to_apples.sh 2>&1 | tee "${TRAIN_LOG}"
