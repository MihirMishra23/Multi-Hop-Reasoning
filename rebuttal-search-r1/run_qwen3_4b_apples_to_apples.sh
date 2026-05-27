#!/bin/bash
#
# Search-R1-style baseline for KBevo apples-to-apples comparison — Qwen3-4B variant.
#
# IDENTICAL recipe to run_qwen3_1.7b_apples_to_apples.sh; only MODEL_PATH +
# EXPERIMENT_NAME change. Same LR, KL, clip, batch shape, rollout count, turns,
# context budget, reward, and step budget — so the only variable in the
# 1.7B-vs-4B comparison is model size.
#
# Hardware note: this run targets aimi-compute-01 (4x B200, 183GB each), but
# the recipe should be reproducible on any node with >=80GB GPUs. The env
# (/scratch/lz586/envs/searchr1-sgl) was rsynced from aimi-02 to keep package
# versions byte-identical.

set -x

ulimit -n 65535

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# CONFIG_PATH / TOOL_CONFIG can be overridden by env so wrappers can point at
# node-local snapshots (protects running jobs from NFS edits/deletes).
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/verl_config}"
TOOL_CONFIG="${TOOL_CONFIG:-${CONFIG_PATH}/tool_config/search_tool_config.yaml}"
REWARD_PATH="${SCRIPT_DIR}/rewards/hotpotqa_f1.py"

TRAIN_DATA="${SCRIPT_DIR}/data/train_verl.parquet"
VAL_DATA="${SCRIPT_DIR}/data/test_verl.parquet"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
NUM_GPUS="${NUM_GPUS:-4}"
LR="${LR:-1e-6}"
LR_TAG="$(echo "${LR}" | sed 's/-//;s/e/e/;s/\.//g')"  # 3e-6 -> 3e6
N_ROLLOUT="${N_ROLLOUT:-5}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-4b-searchR1-a2a-lr${LR_TAG}-n${N_ROLLOUT}-$(date +%Y%m%d_%H%M)}"
# Unique per-launch tag for output dirs (slurm id if available, else timestamp).
RUN_TAG="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"

export WANDB_ENTITY="${WANDB_ENTITY:-ryan-noonan-cornell-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-search-r1-hotpotqa}"

for f in "${TRAIN_DATA}" "${VAL_DATA}" "${TOOL_CONFIG}" "${REWARD_PATH}" \
         "${CONFIG_PATH}/search_multiturn_grpo.yaml"; do
    [ -f "${f}" ] || { echo "MISSING: ${f}"; exit 1; }
done
curl -sf http://127.0.0.1:8000/health > /dev/null \
    || { echo "Retrieval server not responding at :8000 — start launch_retrieval_verl.sh first"; exit 1; }

python3 -m verl.trainer.main_ppo \
    --config-path="${CONFIG_PATH}" \
    --config-name='search_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=16 \
    data.val_batch_size=100 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=cosine \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=${N_ROLLOUT} \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path="${REWARD_PATH}" \
    custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=25 \
    trainer.log_val_generations=10 \
    trainer.rollout_data_dir="${SCRIPT_DIR}/outputs/rollouts_${EXPERIMENT_NAME}_${RUN_TAG}" \
    trainer.total_training_steps=500 \
    trainer.total_epochs=10 $@
