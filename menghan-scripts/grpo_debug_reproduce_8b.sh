#!/bin/bash
# Reproduce the original 8B GRPO run (with the 0.15 reward bug) exactly,
# but log to wandb under a DIFFERENT run name so the original wandb run
# isn't overwritten. Use this to validate that any fix you apply later
# actually moves the needle vs this baseline.
#
# Usage:
#   bash menghan-scripts/grpo_debug_reproduce_8b.sh
#
# Watch for:
#   - train/reward at step 0-5: should be ~0.15-0.18 (matches original)
#   - tools/failure_frequency: ~0.75
#   - [FSDP_TO_VLLM_SYNC] dump for lm_head / embed_tokens
#
# Once you see stable numbers (~20-30 steps), Ctrl+C to stop. No need to run all 500.

set -e

cd /share/j_sun/mx253/Multi-Hop-Reasoning

# Make sure conda env is active
if [[ "$CONDA_DEFAULT_ENV" != "lmlm_b200" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate lmlm_b200
fi

# Unique wandb run name so it doesn't collide with the original run
# (uses date + random so multiple debug runs don't collide either)
export WANDB_NAME="debug-reproduce-8b-$(date +%m%d-%H%M)"
echo "WANDB_NAME=${WANDB_NAME}"

# Same MODEL_PATH as the original GRPO run
MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-8B-SFT_hotpotqa_ep3_bsz48_lr5e-5_th-1

# Save into a debug folder so we don't overwrite the real checkpoints
SAVE_DIR=/share/j_sun/lmlm_multihop/checkpoints/debug

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
echo "Detected ${NUM_GPUS} GPU(s)"

bash scripts/grpo_train.sh \
    --gpu_type B200 \
    --model_path "${MODEL_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --dataset_name hotpotqa \
    --database_path "" \
    --train_size 7000 \
    --total_batch_size 512 \
    --reward_func f1 \
    --phase1_prompt_type sft \
    --phase1_reward_type binary \
    --phase1_db_weight_mode count \
    --num_generations 32 \
    --num_db_rollouts 4 \
    --retrieval_threshold 0.6 \
    --top_k 4 \
    --learning_rate 5e-6 \
    --max_steps 500 \
    --two_phase
