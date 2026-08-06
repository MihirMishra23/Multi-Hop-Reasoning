#!/bin/bash


# SLURM Directives
#SBATCH -J grpo_4b_2ph_0805
#SBATCH -o /home/lz586/icl/slurm_output/lmlm_multihop/grpo_train_4b_2ph_0805_%j.out
#SBATCH -e /home/lz586/icl/slurm_output/lmlm_multihop/grpo_train_4b_2ph_0805_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -t infinite
#SBATCH --partition=aimi,kilian,jjs533,default_partition
#SBATCH --gres=gpu:nvidia_b200:4
#SBATCH --requeue

# Load the required environmentt
source activate mem

# WANDB destination — Menghan's KBevo project
export WANDB_ENTITY=KBevo
export WANDB_PROJECT=LMLM

# Navigate to the working directory
cd /share/j_sun/mx253_lmlm/


# DATASET=hotpotqa-v2
DATASET=hotpotqa-v2

# MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-8B-SFT_hotpotqa_ep3_bsz48_th-1
# MODEL_PATH=/share/j_sun/lmlm_multihop/checkpoints/main/Qwen3-4B-SFT_hotpotqa_ep3_bsz48_th-1/checkpoint-368
# MODEL_PATH=/share/j_sun/rtn27/checkpoints/lmlm_multi_hop/Qwen3-4B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_classic_retrieval_6k
# MODEL_PATH="/share/j_sun/rtn27/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k"
# MODEL_PATH="Qwen/Qwen3-1.7B"
# Mid-training checkpoint (checkpoint-276 unavailable; falling back to 368 per user)
MODEL_PATH=/share/j_sun/lmlm_multihop/checkpoints/main/Qwen3-4B-SFT_hotpotqa_ep3_bsz48_th-1/checkpoint-368
SAVE_DIR=/share/j_sun/lmlm_multihop/checkpoints/main/0805
RETRIEVAL_THRESHOLD=0.6
RETRIEVAL_TOP_K=4
NUM_GENERATIONS=32
NUM_DB_ROLLOUTS=4
DATABASE_PATH=""  # TODO: set a valid database JSON path for single-phase mode

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

# if it is qwen model, then we need to use the zero_rl prompt
USE_CHAT_TEMPLATE=""
if [[ "${MODEL_PATH}" == *"Qwen/Qwen"* ]]; then
    PHASE1_PROMPT_TYPE="formatted_zero_rl"
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
