#!/bin/bash


#SBATCH -J grpo_qwen2.5_7b
#SBATCH -o /home/lz586/icl/slurm_output/lmlm_multihop/grpo_train_qwen2.5_7b_%j.out
#SBATCH -e /home/lz586/icl/slurm_output/lmlm_multihop/grpo_train_qwen2.5_7b_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -t 8:00:00
#SBATCH --partition=aimi
#SBATCH --gres=gpu:nvidia_b200:4              # 4-way FSDP shards the 7B training state
#SBATCH --requeue


# Load the required environment
source activate mem

# Navigate to the working directory
cd /home/lz586/icl/Multi-Hop-Reasoning/


DATASET=hotpotqa-v2

# Mirrors the 8B precedent (checkpoint-368 = mid-training, ~50%).
# 8 checkpoints exist: 92, 184, 276, 368, 460, 552, 644, 735.
MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen2.5-7B-SFT_hotpotqa_ep3_bsz48_lr5e-5_th-1/checkpoint-368

DATABASE_PATH=""  # two-phase mode does not need a static database

THRESHOLD=-1

# Defaults (overridden per dataset below)
REWARD_FUNC="f1"
TWO_PHASE=""
DEBUG=""

export TRITON_CACHE_DIR=/home/lz586/.cache/triton
export TMPDIR=/home/lz586/.tmp
mkdir -p $TRITON_CACHE_DIR $TMPDIR

if [ "${DATASET}" = "hotpotqa-v2" ]; then
    echo "Training on hotpotqa-v2"
    TRAIN_SIZE=7000
    TOTAL_BATCH_SIZE=512
    DATASET_NAME="hotpotqa"
    TWO_PHASE="--two_phase"
elif [ "${DATASET}" = "hotpotqa-v2-onlyphase1" ]; then
    echo "Training on hotpotqa-v2-onlyphase1"
    TRAIN_SIZE=7000
    TOTAL_BATCH_SIZE=512
    DATASET_NAME="hotpotqa"
    TWO_PHASE=""
    DATABASE_PATH=/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_train_start_idx_82347_nb_8100_database.json
elif [ "${DATASET}" = "hotpotqa-v2-debug" ]; then
    echo "Training on hotpotqa-v2-debug"
    TRAIN_SIZE=1000
    TOTAL_BATCH_SIZE=256
    DATASET_NAME="hotpotqa"
    DEBUG="--debug"
    TWO_PHASE="--two_phase"
else
    echo "Invalid dataset: ${DATASET}"
    exit 1
fi

# If the model is a raw HF Qwen checkpoint we need zero-RL prompt + chat template;
# our SFT checkpoint is path-based (no "Qwen/Qwen" prefix), so it uses the sft prompt.
USE_CHAT_TEMPLATE=""
if [[ "${MODEL_PATH}" == *"Qwen/Qwen"* ]]; then
    PHASE1_PROMPT_TYPE="formatted_zero_rl_v5"
    USE_CHAT_TEMPLATE="--use_chat_template"
else
    PHASE1_PROMPT_TYPE="sft"
fi

echo "Starting GRPO Training (Qwen2.5-7B ablation, 4xB200 FSDP)..."
bash /home/lz586/icl/Multi-Hop-Reasoning/scripts/grpo_train.sh \
--gpu_type B200 \
--model_path ${MODEL_PATH} \
--dataset_name ${DATASET_NAME} \
--database_path "${DATABASE_PATH}" \
--train_size ${TRAIN_SIZE} \
--reward_func ${REWARD_FUNC} \
--total_batch_size ${TOTAL_BATCH_SIZE} \
--phase1_prompt_type ${PHASE1_PROMPT_TYPE} \
${USE_CHAT_TEMPLATE} \
${TWO_PHASE} \
${DEBUG}

echo "GRPO Training complete!"
