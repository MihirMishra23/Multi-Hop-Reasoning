#!/bin/bash


# SLURM Directives
#SBATCH -J grpo_train                           # Job name
#SBATCH -o /home/lz586/icl/slurm_output/lmlm_multihop/grpo_train_%j.out   # Standard output file
#SBATCH -e /home/lz586/icl/slurm_output/lmlm_multihop/grpo_train_%j.err   # Standard error file
#SBATCH -N 1                                    # Number of nodes
#SBATCH -n 1                                    # Number of tasks (or cores)
#SBATCH --get-user-env                          # Retrieve the user's login environment
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G                              # Memory requested (2GB per node)
#SBATCH -t infinite                              # Time limit (2 hours)
#SBATCH --partition=aimi,kilian,jjs533,default_partition          # Request partition
#SBATCH --gres=gpu:nvidia_b200:2                # Request 1 NVIDIA A6000 GPU
#SBATCH --requeue


# Load the required environmentt
source activate mem

# Navigate to the working directory
cd /home/lz586/icl/Multi-Hop-Reasoning/


DATASET=hotpotqa-v2
THRESHOLD=-1
DEBUG=""

if [ "${DATASET}" = "hotpotqa" ]; then
    echo "Training on hotpotqa"
    DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_train_start_idx_82347_nb_8100_database.json"
    MODEL_PATH="/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_ep5_bsz48_th${THRESHOLD}"
    TRAIN_SIZE=7000
    DATASET_NAME="hotpotqa"
elif [ "${DATASET}" = "hotpotqa-v1.1" ]; then
    echo "Training on hotpotqa-v1.1"
    DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_train_start_idx_82347_nb_8100_all_context_exp_prompt.json"
    if [ "${THRESHOLD}" = "-1" ]; then
        MODEL_PATH="/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_ep5_bsz48"
    else
        MODEL_PATH="/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_ep5_bsz48_th${THRESHOLD}"
    fi
    TRAIN_SIZE=7000
    DATASET_NAME="hotpotqa"
elif [ "${DATASET}" = "mquake" ]; then
    echo "Training on mquake"
    DATABASE_PATH="/share/j_sun/lmlm_multihop/database/mquake-remastered/mquake_remastered_cf6334_database.json"
    MODEL_PATH="/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_mquake_ep5_bsz48_th${THRESHOLD}"
    TRAIN_SIZE=5334
    DATASET_NAME="mquake"
elif [ "${DATASET}" = "hotpotqa-v2" ]; then
    echo "Training on hotpotqa-v2"
    MODEL_PATH=/share/j_sun/rtn27/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1
    DATABASE_PATH="na" # -> Not used for two phase
    TRAIN_SIZE=7000
    DATASET_NAME="hotpotqa"
    DEBUG=""
    TWO_PHASE="--two_phase"
    REWARD_FUNC="em_size"
elif [ "${DATASET}" = "hotpotqa-v2-debug" ]; then
    echo "Training on hotpotqa-v2-debug"
    MODEL_PATH=/share/j_sun/rtn27/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1
    DATABASE_PATH="na" # -> Not used for two phase
    TRAIN_SIZE=100
    DATASET_NAME="hotpotqa"
    DEBUG="--debug"
    TWO_PHASE="--two_phase"
    REWARD_FUNC="f1_size"
else
    echo "Invalid dataset: ${DATASET}"
    exit 1
fi


if [ ! -d "${MODEL_PATH}" ]; then
    echo "Model path directory does not exist: ${MODEL_PATH}"
    exit 1
fi

echo "Starting GRPO Training..."
bash /home/lz586/icl/Multi-Hop-Reasoning/scripts/grpo_train.sh \
--model_path ${MODEL_PATH} \
--dataset_name ${DATASET_NAME} \
--database_path ${DATABASE_PATH} \
--train_size ${TRAIN_SIZE} \
${DEBUG} \
${TWO_PHASE} \
--reward_func ${REWARD_FUNC}

echo "GRPO Training complete!"
