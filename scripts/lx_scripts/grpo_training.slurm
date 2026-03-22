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
#SBATCH --gres=gpu:nvidia_b200:1                # Request 1 NVIDIA A6000 GPU
#SBATCH --requeue


# Load the required environmentt
source activate mem

# Navigate to the working directory
cd /home/lz586/icl/Multi-Hop-Reasoning/


DATASET=hotpotqa-v2

MODEL_PATH="/share/j_sun/rtn27/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k"
DATABASE_PATH=""  # Not used in two-phase mode

THRESHOLD=-1

# Defaults (overridden per dataset below)
REWARD_FUNC="em"


TWO_PHASE=""
DEBUG=""


if [ "${DATASET}" = "hotpotqa-v2" ]; then
    echo "Training on hotpotqa-v2"
    TRAIN_SIZE=7000
    TOTAL_BATCH_SIZE=512
    DATASET_NAME="hotpotqa"
    TWO_PHASE="--two_phase"
elif [ "${DATASET}" = "hotpotqa-v2-debug" ]; then
    echo "Training on hotpotqa-v2-debug"
    TRAIN_SIZE=1000
    TOTAL_BATCH_SIZE=2048
    DATASET_NAME="hotpotqa"
    DEBUG="--debug"
    TWO_PHASE="--two_phase"
else
    echo "Invalid dataset: ${DATASET}"
    exit 1
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
${TWO_PHASE} \
${DEBUG}

echo "GRPO Training complete!"
