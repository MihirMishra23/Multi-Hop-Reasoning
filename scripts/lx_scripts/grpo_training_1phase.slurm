#!/bin/bash


# SLURM Directives
#SBATCH -J grpo_1.7b_1ph_0805
#SBATCH -o /home/lz586/icl/slurm_output/lmlm_multihop/grpo_train_1.7b_1ph_0805_%j.out
#SBATCH -e /home/lz586/icl/slurm_output/lmlm_multihop/grpo_train_1.7b_1ph_0805_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --get-user-env
#SBATCH --cpus-per-task=28
#SBATCH --mem=256G
#SBATCH -t infinite
#SBATCH --partition=aimi,kilian,jjs533,default_partition
#SBATCH --gres=gpu:nvidia_b200:4
#SBATCH --requeue


# Load the required environment
source activate mem

# WANDB destination — Menghan's KBevo project
export WANDB_ENTITY=KBevo
export WANDB_PROJECT=LMLM

# Navigate to the working directory
cd /share/j_sun/mx253_lmlm/


DATASET=hotpotqa-v2-onlyphase1

# 1-phase SFT base (matches historical canonical 1-phase GRPO run in main/)
# NOTE: 'onlyphase1' is a misleading legacy name — it actually means
# "phase-2-only GRPO on top of a 1-phase SFT with a pre-built static DB".
# The DB comes from Gemini extractions, not phase-1 rollouts.
MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_1phase
SAVE_DIR=/share/j_sun/lmlm_multihop/checkpoints/main/0805

RETRIEVAL_THRESHOLD=0.6
RETRIEVAL_TOP_K=4
NUM_GENERATIONS=32
NUM_DB_ROLLOUTS=4

DATABASE_PATH=""  # set per-dataset below (phase-2-only mode)

THRESHOLD=-1

# Defaults (overridden per dataset below)
REWARD_FUNC="f1"
TWO_PHASE=""
DEBUG=""


if [ "${DATASET}" = "hotpotqa-v2" ]; then
    echo "Training on hotpotqa-v2"
    TRAIN_SIZE=7000
    TOTAL_BATCH_SIZE=512
    DATASET_NAME="hotpotqa"
    TWO_PHASE="--two_phase"
elif [ "${DATASET}" = "hotpotqa-v2-onlyphase1" ]; then
    echo "Training on hotpotqa-v2-onlyphase1 (phase-2-only + Gemini DB)"
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

# if it is qwen model, then we need to use the zero_rl prompt
USE_CHAT_TEMPLATE=""
if [[ "${MODEL_PATH}" == *"Qwen/Qwen"* ]]; then
    PHASE1_PROMPT_TYPE="formatted_zero_rl_v5"
    USE_CHAT_TEMPLATE="--use_chat_template"
else
    PHASE1_PROMPT_TYPE="sft"
fi

echo "Starting GRPO Training..."
bash /share/j_sun/mx253_lmlm/scripts/grpo_train.sh \
--gpu_type B200 \
--model_path ${MODEL_PATH} \
--save_dir ${SAVE_DIR} \
--dataset_name ${DATASET_NAME} \
--database_path "${DATABASE_PATH}" \
--train_size ${TRAIN_SIZE} \
--reward_func ${REWARD_FUNC} \
--total_batch_size ${TOTAL_BATCH_SIZE} \
--phase1_prompt_type ${PHASE1_PROMPT_TYPE} \
${USE_CHAT_TEMPLATE} \
${TWO_PHASE} \
${DEBUG} \
--retrieval_threshold ${RETRIEVAL_THRESHOLD} \
--retrieval_top_k ${RETRIEVAL_TOP_K} \
--num_generations ${NUM_GENERATIONS} \
--num_db_rollouts ${NUM_DB_ROLLOUTS} \
--use_inverses

echo "GRPO Training complete!"
