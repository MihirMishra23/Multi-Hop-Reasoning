#!/bin/bash

#SBATCH -J grpo_1.7b_tier
#SBATCH -o /home/lz586/icl/slurm_output/lmlm_multihop/grpo_train_1.7b_tier_%j.out
#SBATCH -e /home/lz586/icl/slurm_output/lmlm_multihop/grpo_train_1.7b_tier_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=28
#SBATCH --mem=256G
#SBATCH -t 3-00:00:00
#SBATCH --partition=aimi
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --requeue


# Load the required environmentt
source activate mem

# Navigate to the working directory
cd /home/lz586/icl/Multi-Hop-Reasoning/


DATASET=hotpotqa-v2

# MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-8B-SFT_hotpotqa_ep3_bsz48_th-1
# MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-4B-SFT_hotpotqa_ep3_bsz48_th-1
# MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_1phase 
# MODEL_PATH=/share/j_sun/lmlm_multihop/checkpoints/main/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.9_top_k_1
# MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1
MODEL_PATH=/share/j_sun/lmlm_multihop/checkpoints/main/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.6_top_k_4
RETRIEVAL_THRESHOLD=0.6
TOP_K=4
NUM_GENERATIONS=32
NUM_DB_ROLLOUTS=4

# Tier prefiltering: keep examples whose rollout success score is in [TIER_MIN, TIER_MAX]
TIER_PATH=/share/j_sun/lmlm_multihop/output_tier/hotpotqa_train_lmlm_n7000_i83347_v1.json
TIER_MIN_SCORE=1
TIER_MAX_SCORE=7

# Natural curriculum: expand pool from easy (scores 5-7) → medium (3-7) → hard (1-7)
# Phase boundaries at 33% and 67% of max_steps
CURRICULUM=""
# CURRICULUM="--curriculum"
# CURRICULUM_PHASES="5-7,3-7,1-7"
# CURRICULUM_STEPS="0.33,0.67"

export TRITON_CACHE_DIR=/home/lz586/.cache/triton
export TMPDIR=/home/lz586/.tmp
mkdir -p $TRITON_CACHE_DIR $TMPDIR

# MODEL_PATH=/share/j_sun/rtn27/checkpoints/lmlm_multi_hop/Qwen3-4B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_classic_retrieval_6k
# MODEL_PATH="/share/j_sun/rtn27/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k"
# MODEL_PATH="Qwen/Qwen3-1.7B"
DATABASE_PATH=""  # TODO: set a valid database JSON path for single-phase mode

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

# if it is qwen model, then we need to use the zero_rl prompt
USE_CHAT_TEMPLATE=""
if [[ "${MODEL_PATH}" == *"Qwen/Qwen"* ]]; then
    PHASE1_PROMPT_TYPE="formatted_zero_rl_v5"
    USE_CHAT_TEMPLATE="--use_chat_template"
else
    PHASE1_PROMPT_TYPE="sft"
fi

echo "Starting GRPO Training..."
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
    ${DEBUG} \
    --retrieval_threshold ${RETRIEVAL_THRESHOLD} \
    --top_k ${TOP_K} \
    --num_generations ${NUM_GENERATIONS} \
    --num_db_rollouts ${NUM_DB_ROLLOUTS} \
    --tier_path ${TIER_PATH} \
    --tier_min_score ${TIER_MIN_SCORE} \
    --tier_max_score ${TIER_MAX_SCORE} \
    ${CURRICULUM} \
    --curriculum_phases ${CURRICULUM_PHASES} \
    --curriculum_steps ${CURRICULUM_STEPS}

echo "GRPO Training complete!"
