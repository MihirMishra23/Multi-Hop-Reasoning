#!/bin/bash
#
# sbatch wrapper for Search-R1 a2a runs (1.7B by default; override LAUNCHER for 4B).
#
# Submit with:
#   LR=3e-6 sbatch --nodelist=aimi-compute-03 sbatch_qwen3_1.7b_apples_to_apples.slurm
#   LAUNCHER=run_qwen3_4b_apples_to_apples.sh sbatch ... (for the 4B variant)
#
# Self-contained:
#   - ensures env is on the target node's /scratch (rsync from aimi-02 if missing)
#   - snapshots verl_config to node-local /scratch so a future NFS edit/delete
#     can't reach the running job (root cause of the 2026-05-27 01:03 crash)
#   - launches a local retrieval server on :8000
#   - runs the chosen launcher
#   - cleans up retrieval on exit even if training crashes

#SBATCH -J searchr1_a2a_1.7b
#SBATCH -o /home/lz586/icl/slurm_output/searchr1_a2a/%x_%j.out
#SBATCH -e /home/lz586/icl/slurm_output/searchr1_a2a/%x_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=1000G
#SBATCH -t 10:00:00
#SBATCH --partition=aimi
#SBATCH --gres=gpu:nvidia_b200:4

set -euo pipefail

LR="${LR:-1e-6}"
LAUNCHER="${LAUNCHER:-run_qwen3_1.7b_apples_to_apples.sh}"
SOURCE_NODE="${SOURCE_NODE:-aimi-compute-02}"
ENV_DIR="/scratch/lz586/envs/searchr1-sgl"
PROJECT_DIR="/home/lz586/icl/Multi-Hop-Reasoning/rebuttal-search-r1"
LOCAL_RUN_DIR="/scratch/lz586/run_${SLURM_JOB_ID}"
LOCAL_CONFIG_PATH="${LOCAL_RUN_DIR}/verl_config"
RETRIEVAL_LOG="${PROJECT_DIR}/logs/retrieval_sbatch_${SLURM_JOB_ID}.log"
TRAIN_LOG="${PROJECT_DIR}/logs/train_a2a_lr${LR}_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M).log"

# Redirect Ray's spilled-objects dir to /scratch (~100GB+) instead of /tmp (15GB).
# Without this, long-running multi-turn rollouts exhaust /tmp on the compute node
# and Ray dies with "OSError: [Errno 28] No space left on device" (hit by 969064
# at step 81 on 2026-05-31). RAY_TMPDIR is honored by Ray on startup.
export RAY_TMPDIR="/scratch/lz586/ray_tmp/${SLURM_JOB_ID}"
mkdir -p "${RAY_TMPDIR}"
echo "[$(date)] RAY_TMPDIR=${RAY_TMPDIR}"

# Same /tmp-exhaustion class of failure killed 971203 (4B) on aimi-compute-02 at
# 2026-06-01 04:21: Triton's JIT kernel cache (default /tmp/triton_cache_$USER)
# ran out of space during sglang model compile. Inductor's cache (/var/tmp by
# default) is in the same boat. Redirect both to /scratch.
export TRITON_CACHE_DIR="/scratch/lz586/triton_cache/${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="/scratch/lz586/torchinductor/${SLURM_JOB_ID}"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
echo "[$(date)] TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
echo "[$(date)] TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}"

# Override TMPDIR — sbatch via --export=ALL inherits whatever TMPDIR the submitting
# shell had. When the submitter is Claude Code, that is /tmp/claude-$UID, which is
# tiny and shared. With 8 sglang workers concurrently calling gcc to compile
# cuda_utils.c via tempfile.mkdtemp() (which honors TMPDIR), /tmp exhausts and
# sglang actors SIGABRT during startup (hit by 971737 on aimi-compute-02 at
# 2026-06-01 12:40: "Cannot create temporary file in /tmp/claude-1733861/").
export TMPDIR="/scratch/lz586/tmp/${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"
echo "[$(date)] TMPDIR=${TMPDIR}"

echo "[$(date)] === sbatch start ==="
echo "Node:      $(hostname)"
echo "JobID:     ${SLURM_JOB_ID}"
echo "LR:        ${LR}"
echo "GPUs:      $(nvidia-smi -L | wc -l) visible"
nvidia-smi -L

# ----- 1. Env: ensure searchr1-sgl is on /scratch -----
if [ ! -x "${ENV_DIR}/bin/python" ]; then
    echo "[$(date)] Env missing on this node — rsync from ${SOURCE_NODE}"
    mkdir -p /scratch/lz586/envs
    rsync -a --no-perms "${SOURCE_NODE}:${ENV_DIR}/" "${ENV_DIR}/"
fi
export PATH="${ENV_DIR}/bin:${PATH}"
echo "[$(date)] python: $(which python3)"
python3 --version

# ----- 1a. Ensure agent_loop.py has NO </answer> stop string -----
# An earlier experiment patched stop=["</answer>"] into the sampling_params dict,
# but that path requires sglang's internal tokenizer (incompatible with verl's
# skip_tokenizer_init=True). We reverted the approach. This sed removes the line
# if a prior patched env is reused; it's a no-op on a clean env.
AGENT_LOOP_FILE="${ENV_DIR}/lib/python3.11/site-packages/verl/experimental/agent_loop/agent_loop.py"
if grep -q 'stop=\["</answer>"\]' "${AGENT_LOOP_FILE}" 2>/dev/null; then
    echo "[$(date)] Removing stale stop=[\"</answer>\"] from ${AGENT_LOOP_FILE}"
    sed -i '/stop=\["<\/answer>"\],/d' "${AGENT_LOOP_FILE}"
fi

# ----- 1b. Snapshot verl_config to node-local /scratch -----
# verl re-reads tool_config every ToolAgentLoop instantiation. If the NFS
# source is edited/deleted mid-run, all readers crash. Snapshotting to
# node-local /scratch isolates running jobs from NFS hiccups.
echo "[$(date)] Snapshotting verl_config -> ${LOCAL_CONFIG_PATH}"
mkdir -p "${LOCAL_RUN_DIR}"
cp -r "${PROJECT_DIR}/verl_config" "${LOCAL_CONFIG_PATH}"
ls "${LOCAL_CONFIG_PATH}/tool_config/" || { echo "[$(date)] snapshot failed — abort"; exit 1; }
export CONFIG_PATH="${LOCAL_CONFIG_PATH}"
export TOOL_CONFIG="${LOCAL_CONFIG_PATH}/tool_config/search_tool_config.yaml"

# ----- 2. Retrieval server (background, GPU 0 for the E5 encoder) -----
# If another sbatch on this node already launched retrieval, reuse it. Use flock
# to serialize the check+launch decision so two co-located jobs don't both bind
# port 8000. We do NOT kill retrieval in cleanup if another job is still using it.
cd "${PROJECT_DIR}"
mkdir -p logs outputs
RETRIEVAL_LAUNCHED_BY_US=0
RETRIEVAL_PID=""
# Lock and status files live in /scratch (per-node fast local) instead of /tmp,
# because some compute nodes have /tmp exhausted by other users (hit by 971195
# on aimi-compute-02 at 2026-06-01: "No space left on device" writing a few-byte
# status file). LOCK_FILE must be shared across co-located jobs on the same node,
# so it lives in the user-level /scratch dir; status file is per-job in LOCAL_RUN_DIR.
SCRATCH_USER_DIR="/scratch/lz586"
mkdir -p "${SCRATCH_USER_DIR}"
LOCK_FILE="${SCRATCH_USER_DIR}/retrieval_${USER}.lock"
RETRIEVAL_STATUS_FILE="${LOCAL_RUN_DIR}/retrieval_status.txt"

# Fast path: skip the flock entirely if retrieval is already up. This avoids the
# fd-inheritance bug where the launched retrieval server inherits fd 200 and
# holds the lock forever, blocking later co-located jobs (hit by 967925 on
# 2026-05-31). On a clean env or first-on-node, we still enter the flock branch.
if curl -sf -m 2 http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "[$(date)] retrieval already up, reusing"
    echo "RETRIEVAL_LAUNCHED_BY_US=0" > "${RETRIEVAL_STATUS_FILE}"
else
    (
        exec 200>"${LOCK_FILE}"
        flock 200
        if curl -sf -m 2 http://127.0.0.1:8000/health > /dev/null 2>&1; then
            echo "RETRIEVAL_LAUNCHED_BY_US=0"
        else
            echo "RETRIEVAL_LAUNCHED_BY_US=1"
            # Close lock fd in the child + setsid so retrieval server doesn't
            # inherit fd 200 (which would hold the lock forever).
            (
                exec 200>&-
                CUDA_VISIBLE_DEVICES=0 setsid bash launch_retrieval_verl.sh \
                    > "${RETRIEVAL_LOG}" 2>&1 < /dev/null &
                disown
            )
            for i in $(seq 1 60); do
                if curl -sf -m 2 http://127.0.0.1:8000/health > /dev/null 2>&1; then
                    break
                fi
                sleep 1
            done
        fi
    ) > "${RETRIEVAL_STATUS_FILE}"
fi

eval "$(cat "${RETRIEVAL_STATUS_FILE}")"
echo "[$(date)] Retrieval state: launched_by_us=${RETRIEVAL_LAUNCHED_BY_US} pid=${RETRIEVAL_PID:-none}"

# Final health gate
curl -sf -m 5 http://127.0.0.1:8000/health > /dev/null \
    || { echo "[$(date)] retrieval never became healthy — abort"; tail -50 "${RETRIEVAL_LOG}" 2>/dev/null; exit 1; }
echo "[$(date)] retrieval is healthy"

cleanup() {
    # Don't kill retrieval — another sbatch on this node may still be using it.
    # Manual cleanup via `pkill -f retrieval_server_verl` if needed.
    rm -rf "${LOCAL_RUN_DIR}" 2>/dev/null || true
    # Status file lives inside LOCAL_RUN_DIR now, removed by the rm -rf above.
    :
}
trap cleanup EXIT

# ----- 3. Training (foreground; this is the actual job) -----
# Optional resume: if RESUME_PATH is set, append the resume overrides as extra
# args. The launcher passes "$@" through to verl, so trailing hydra overrides
# work. EXPERIMENT_NAME (if set) is exported so the launcher picks it up and
# writes into the original checkpoint dir.
RESUME_ARGS=()
if [ -n "${RESUME_PATH:-}" ]; then
    RESUME_ARGS+=("trainer.resume_mode=resume_path" "trainer.resume_from_path=${RESUME_PATH}")
    echo "[$(date)] Resuming from: ${RESUME_PATH}"
fi
EXTRA_ARGS_ARR=()
if [ -n "${EXTRA_ARGS:-}" ]; then
    # Caller can pass extra hydra overrides via env var EXTRA_ARGS.
    # Space-separated; each token becomes one arg. Use single-token overrides
    # (no embedded spaces) — e.g. EXTRA_ARGS="+data.foo=bar trainer.x=1"
    read -ra EXTRA_ARGS_ARR <<< "${EXTRA_ARGS}"
    echo "[$(date)] Extra hydra args: ${EXTRA_ARGS}"
fi
echo "[$(date)] Launching training with LAUNCHER=${LAUNCHER} LR=${LR} EXPERIMENT_NAME=${EXPERIMENT_NAME:-<default>} -> ${TRAIN_LOG}"
LR="${LR}" EXPERIMENT_NAME="${EXPERIMENT_NAME:-}" \
TRAIN_DATA="${TRAIN_DATA:-}" VAL_DATA="${VAL_DATA:-}" \
CONFIG_PATH="${CONFIG_PATH}" TOOL_CONFIG="${TOOL_CONFIG}" \
    bash "${LAUNCHER}" "${RESUME_ARGS[@]}" "${EXTRA_ARGS_ARR[@]}" 2>&1 | tee "${TRAIN_LOG}"
TRAIN_EXIT=${PIPESTATUS[0]}

echo "[$(date)] === sbatch end (train exit=${TRAIN_EXIT}) ==="
exit "${TRAIN_EXIT}"
