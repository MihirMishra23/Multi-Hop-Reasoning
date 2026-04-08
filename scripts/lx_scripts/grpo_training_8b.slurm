#!/bin/bash


#SBATCH -J grpo_8b_optimized
#SBATCH -o /home/lz586/icl/slurm_output/lmlm_multihop/grpo_train_8b_optimized_%j.out
#SBATCH -e /home/lz586/icl/slurm_output/lmlm_multihop/grpo_train_8b_optimized_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH -t 3-00:00:00
#SBATCH --partition=aimi
#SBATCH --gres=gpu:nvidia_b200:4
#SBATCH --requeue


# Load the required environmentt
source activate mem

# Navigate to the working directory
cd /home/lz586/icl/Multi-Hop-Reasoning/


# DATASET=hotpotqa-v2
DATASET=hotpotqa-v2

MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-8B-SFT_hotpotqa_ep3_bsz48_lr5e-5_th-1
# MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-8B-SFT_hotpotqa_ep3_bsz48_th-1
# MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-4B-SFT_hotpotqa_ep3_bsz48_th-1
# MODEL_PATH=/share/j_sun/rtn27/checkpoints/lmlm_multi_hop/Qwen3-4B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_classic_retrieval_6k
# MODEL_PATH="/share/j_sun/rtn27/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k"
# MODEL_PATH="Qwen/Qwen3-1.7B"
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
bash /home/lz586/icl/Multi-Hop-Reasoning/scripts/lx_scripts/grpo_train.sh \
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
