#!/bin/bash
# Debug script for FSDP→vLLM weight sync bug on 8B model.
# Based on scripts/lx_scripts/grpo_training_8b.slurm, adapted for mx253.
# Goal: trigger _sync_fsdp1_params_to_vllm() and observe its debug prints.
#
# We only need step 0 — once you see lm_head/embed_tokens lines in the .err log,
# you can cancel the job (scancel <JOBID>).

#SBATCH --job-name=grpo-debug-8b-sync
#SBATCH --partition=aimi,default_partition
#SBATCH --gres=gpu:nvidia_b200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/grpo_debug_8b_sync_%j.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/grpo_debug_8b_sync_%j.err

eval "$(conda shell.bash hook)"
conda activate lmlm_b200

cd /share/j_sun/mx253/Multi-Hop-Reasoning

DATASET_NAME="hotpotqa"
MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-8B-SFT_hotpotqa_ep3_bsz48_lr5e-5_th-1
DATABASE_PATH=""

# SFT-style prompt (Qwen3 SFT model, not raw zero-shot)
PHASE1_PROMPT_TYPE="sft"

# Small for fast iteration; we only need step 0 to fire the sync.
TRAIN_SIZE=100
TOTAL_BATCH_SIZE=128
REWARD_FUNC="f1"

echo "Starting 8B GRPO (debug sync run)..."
bash scripts/grpo_train.sh \
    --gpu_type B200 \
    --model_path "${MODEL_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --database_path "${DATABASE_PATH}" \
    --train_size ${TRAIN_SIZE} \
    --reward_func ${REWARD_FUNC} \
    --total_batch_size ${TOTAL_BATCH_SIZE} \
    --phase1_prompt_type ${PHASE1_PROMPT_TYPE} \
    --two_phase

echo "Run complete (check .err for [FSDP→vLLM SYNC] lines)."
