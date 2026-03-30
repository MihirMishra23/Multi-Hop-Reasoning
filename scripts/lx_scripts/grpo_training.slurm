#!/bin/bash


# SLURM Directives
#SBATCH -J grpo_1.7b_2phase_ret_0.9_top_k_1                   # Job name
#SBATCH -o /home/lz586/icl/slurm_output/lmlm_multihop/grpo_train_1.7b_2phase_ret_0.9_top_k_1_%j.out   # Standard output file
#SBATCH -e /home/lz586/icl/slurm_output/lmlm_multihop/grpo_train_1.7b_2phase_ret_0.9_top_k_1_%j.err   # Standard error file
#SBATCH -N 1                                    # Number of nodes
#SBATCH -n 1                                    # Number of tasks (or cores)
#SBATCH --get-user-env                          # Retrieve the user's login environment
#SBATCH --cpus-per-task=28
#SBATCH --mem=256G                              # Memory requested (2GB per node)
#SBATCH -t infinite                              # Time limit (2 hours)
#SBATCH --partition=aimi,kilian,jjs533,default_partition          # Request partition
#SBATCH --gres=gpu:nvidia_b200:1                # Request 1 NVIDIA A6000 GPU
#SBATCH --requeue
#SBATCH --nodelist=aimi-compute-01,aimi-compute-02,aimi-compute-03


# Load the required environmentt
source activate mem

# Navigate to the working directory
cd /home/lz586/icl/Multi-Hop-Reasoning/


DATASET=hotpotqa-v2

# MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-8B-SFT_hotpotqa_ep3_bsz48_th-1
# MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-4B-SFT_hotpotqa_ep3_bsz48_th-1
MODEL_PATH=Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.9_top_k_1
RETRIEVAL_THRESHOLD=0.9
TOP_K=1
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
${DEBUG} \
--retrieval_threshold ${RETRIEVAL_THRESHOLD} \
--top_k ${TOP_K}

echo "GRPO Training complete!"
