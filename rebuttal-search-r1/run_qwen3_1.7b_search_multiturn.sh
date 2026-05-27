#!/bin/bash
#
# Search-R1-like multi-turn RL on Qwen3-1.7B + HotpotQA, using upstream
# verl 0.7.0's recipe (examples/sglang_multiturn/search_r1_like/), adapted
# for our 4×B200 box (aimi-compute-02) and local HotpotQA corpus.
#
# Prereqs:
#   1. Env: /scratch/lz586/envs/searchr1-sgl  (sglang + verl + faiss-cpu)
#   2. Retrieval server running on http://127.0.0.1:8000/retrieve
#        (see launch_retrieval_verl.sh)
#   3. Adapted parquets at data/{train,test}_verl.parquet
#
# Launch:
#   bash run_qwen3_1.7b_search_multiturn.sh \
#     trainer.experiment_name=qwen3-1.7b-searchR1-like-$(date +%Y%m%d_%H%M)

set -x

ulimit -n 65535

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/verl_config"
TOOL_CONFIG="${CONFIG_PATH}/tool_config/search_tool_config.yaml"

TRAIN_DATA="${SCRIPT_DIR}/data/train_verl.parquet"
VAL_DATA="${SCRIPT_DIR}/data/test_verl.parquet"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
NUM_GPUS="${NUM_GPUS:-4}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-1.7b-searchR1-like-$(date +%Y%m%d_%H%M)}"

export WANDB_PROJECT="${WANDB_PROJECT:-search-r1-hotpotqa-rebuttal}"

# Sanity
for f in "${TRAIN_DATA}" "${VAL_DATA}" "${TOOL_CONFIG}" "${CONFIG_PATH}/search_multiturn_grpo.yaml"; do
    [ -f "${f}" ] || { echo "MISSING: ${f}"; exit 1; }
done
curl -sf http://127.0.0.1:8000/health > /dev/null \
    || { echo "Retrieval server not responding at :8000 — start launch_retrieval_verl.sh first"; exit 1; }

python3 -m verl.trainer.main_ppo \
    --config-path="${CONFIG_PATH}" \
    --config-name='search_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=256 \
    data.val_batch_size=100 \
    data.max_prompt_length=4096 \
    data.max_response_length=3000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_model_len=15000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 $@
