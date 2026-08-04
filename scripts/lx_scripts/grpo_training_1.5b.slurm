#!/bin/bash


# SLURM Directives
#SBATCH -J grpo_qwen2.5_1.5b                       # Job name
#SBATCH -o /home/lz586/icl/slurm_output/lmlm_multihop/grpo_train_qwen2.5_1.5b_%j.out
#SBATCH -e /home/lz586/icl/slurm_output/lmlm_multihop/grpo_train_qwen2.5_1.5b_%j.err
#SBATCH -N 1                                     # Number of nodes
#SBATCH -n 1                                     # Number of tasks (or cores)
#SBATCH --get-user-env
#SBATCH --cpus-per-task=28
#SBATCH --mem=256G
#SBATCH -t infinite
#SBATCH --partition=aimi,kilian,jjs533,default_partition
#SBATCH --gres=gpu:nvidia_b200:4                 # 4x B200 for throughput (1.5B fits on a single GPU)
#SBATCH --requeue
#SBATCH --nodelist=aimi-compute-01,aimi-compute-02,aimi-compute-03


# Load the required environment
source activate mem

# Navigate to the working directory
cd /home/lz586/icl/Multi-Hop-Reasoning/


DATASET=hotpotqa-v2

# Qwen2.5-1.5B SFT checkpoint (effective bsz=48, lr=5e-5, threshold=-1, two-phase).
# Append /checkpoint-XXX to lock to a specific step instead of latest.
MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen2.5-1.5B-SFT_hotpotqa_ep3_bsz48_lr5e-5_th-1
# MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen2.5-1.5B-SFT_hotpotqa_ep3_bsz48_lr5e-5_th-1/checkpoint-XXX

RETRIEVAL_THRESHOLD=0.6
TOP_K=4
NUM_GENERATIONS=32
NUM_DB_ROLLOUTS=4

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

echo "Starting GRPO Training (Qwen2.5-1.5B ablation)..."
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
--num_db_rollouts ${NUM_DB_ROLLOUTS}

echo "GRPO Training complete!"
