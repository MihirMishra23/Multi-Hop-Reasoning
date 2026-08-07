#!/usr/bin/env bash
# Shared launcher for Linxi's May 2026 Search-R1 Qwen3 paper runs (c30a5d9).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/verl_config}"
TOOL_CONFIG="${TOOL_CONFIG:-${CONFIG_PATH}/tool_config/search_tool_config.yaml}"
REWARD_PATH="${REWARD_PATH:-${SCRIPT_DIR}/rewards/hotpotqa_f1.py}"

TRAIN_DATA="${TRAIN_DATA:-${SCRIPT_DIR}/data/train_verl.parquet}"
VAL_DATA="${VAL_DATA:-${SCRIPT_DIR}/data/test_verl.parquet}"
MODEL_ARTIFACT="${MODEL_ARTIFACT:-qwen3_1_7b}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_REVISION="${MODEL_REVISION:-}"
MODEL_RUN_NAME="${MODEL_RUN_NAME:-qwen3-1.7b}"
NUM_GPUS="${NUM_GPUS:-4}"
LR="${LR:-3e-6}"
N_ROLLOUT="${N_ROLLOUT:-8}"
RETRIEVAL_URL="${RETRIEVAL_URL:-http://127.0.0.1:8000/retrieve}"
RETRIEVAL_HEALTH_URL="${RETRIEVAL_HEALTH_URL:-${RETRIEVAL_URL%/retrieve}/health}"
PYTHON="${PYTHON:-python3}"

LR_TAG="$(printf '%s' "${LR}" | sed 's/-//;s/\.//g')"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL_RUN_NAME}-searchR1-a2a-lr${LR_TAG}-n${N_ROLLOUT}-$(date +%Y%m%d_%H%M)}"
RUN_TAG="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${SCRIPT_DIR}/outputs/rollouts_${EXPERIMENT_NAME}_${RUN_TAG}}"

export WANDB_ENTITY="${WANDB_ENTITY:-KBevo}"
export WANDB_PROJECT="${WANDB_PROJECT:-LMLM}"
export RETRIEVAL_URL

# Keep this pinned environment isolated from packages installed in ~/.local.
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
# The CUDA 13 DeepGEMM wheel currently selected by SGLang's optional extra
# cannot load on the CUDA 12.8 paper hardware. These runs use BF16, not FP8.
export SGL_ENABLE_JIT_DEEPGEMM="${SGL_ENABLE_JIT_DEEPGEMM:-false}"

if [ -z "${MODEL_PATH}" ]; then
    MODEL_PATH="$("${PYTHON}" "${SCRIPT_DIR}/reproduction_manifest.py" --artifact "${MODEL_ARTIFACT}")"
elif [ -d "${MODEL_PATH}" ]; then
    MODEL_PATH="$(cd "${MODEL_PATH}" && pwd)"
elif [ -n "${MODEL_REVISION}" ]; then
    MODEL_PATH="$("${PYTHON}" "${SCRIPT_DIR}/reproduction_manifest.py" \
        --repo-id "${MODEL_PATH}" --revision "${MODEL_REVISION}")"
else
    echo "A remote MODEL_PATH requires a full MODEL_REVISION commit hash." >&2
    exit 2
fi

echo "Reproduction provenance: $("${PYTHON}" "${SCRIPT_DIR}/reproduction_manifest.py" --describe)"
echo "Resolved model snapshot: ${MODEL_PATH}"

for path in \
    "${TRAIN_DATA}" \
    "${VAL_DATA}" \
    "${TOOL_CONFIG}" \
    "${REWARD_PATH}" \
    "${CONFIG_PATH}/search_multiturn_grpo.yaml"; do
    [ -f "${path}" ] || { echo "Missing input: ${path}" >&2; exit 1; }
done

curl -fsS --max-time 5 "${RETRIEVAL_HEALTH_URL}" >/dev/null || {
    echo "Retrieval service is not healthy at ${RETRIEVAL_HEALTH_URL}" >&2
    exit 1
}

ulimit -n 65535 2>/dev/null || echo "Warning: could not raise the open-file limit" >&2
mkdir -p "${ROLLOUT_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/checkpoints/${WANDB_PROJECT}/${EXPERIMENT_NAME}}"

RAY_ARGS=()
# Ray otherwise sees every host CPU rather than the cores granted by Slurm.
RAY_CPU_COUNT="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-}}"
if [ -n "${RAY_CPU_COUNT}" ]; then
    case "${RAY_CPU_COUNT}" in
        *[!0-9]*|0) echo "RAY_NUM_CPUS must be a positive integer" >&2; exit 2 ;;
    esac
    RAY_ARGS+=("ray_kwargs.ray_init.num_cpus=${RAY_CPU_COUNT}")
fi

"${PYTHON}" -m verl.trainer.main_ppo \
    --config-path="${CONFIG_PATH}" \
    --config-name=search_multiturn_grpo \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    actor_rollout_ref.rollout.n="${N_ROLLOUT}" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
    custom_reward_function.path="${REWARD_PATH}" \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.rollout_data_dir="${ROLLOUT_DIR}" \
    "${RAY_ARGS[@]}" \
    "$@"
