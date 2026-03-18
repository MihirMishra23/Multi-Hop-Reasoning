#!/bin/bash


# Load the required environmentt
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lmlm

# Navigate to the working directory
cd /home/dg793/Multi-Hop-Reasoning/


DATASET=hotpotqa-v2-debug
THRESHOLD=-1
DEBUG=""
PHASE1_REWARD_TYPE="binary"  # "binary" or "utilization" (used_triplets/total_triplets)
PHASE1_PROMPT_TYPE="context_only"  # "context_only" or "with_question"
NUM_DB_ROLLOUTS=1  # K: DB rollouts per question (N must be divisible by K)
NUM_DB_ROLLOUTS_SET=0  # tracks whether --num_db_rollouts was explicitly passed
PHASE1_DB_WEIGHT_MODE="count_dynamic"  # none | fixed[_<w>] | dynamic | count | count_dynamic
LEARNING_RATE=""
RETRIEVAL_THRESHOLD_ARG=""
TOP_K_ARG=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --phase1_reward_type)
            PHASE1_REWARD_TYPE="$2"
            shift 2
            ;;
        --phase1_prompt_type)
            PHASE1_PROMPT_TYPE="$2"
            shift 2
            ;;
        --phase1_db_weight_mode)
            PHASE1_DB_WEIGHT_MODE="$2"
            shift 2
            ;;
        --num_db_rollouts)
            NUM_DB_ROLLOUTS="$2"
            NUM_DB_ROLLOUTS_SET=1
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="--learning_rate $2"
            shift 2
            ;;
        --threshold)
            RETRIEVAL_THRESHOLD_ARG="--retrieval-threshold $2"
            shift 2
            ;;
        --top_k)
            TOP_K_ARG="--top_k $2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

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
    MODEL_PATH=/share/j_sun/rtn27/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_two_phase_hotpotqa_ep5_bsz48
    MODEL_PATH=/share/j_sun/rtn27/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed
    DATABASE_PATH="na" # -> Not used for two phase
    TRAIN_SIZE=7000
    DATASET_NAME="hotpotqa"
    DEBUG=""
    TWO_PHASE="--two_phase"
    REWARD_FUNC="em_size"
    if [ "${NUM_DB_ROLLOUTS_SET}" = "0" ]; then NUM_DB_ROLLOUTS=4; fi
elif [ "${DATASET}" = "hotpotqa-v2-debug" ]; then
    echo "Training on hotpotqa-v2-debug"
    # MODEL_PATH=/share/j_sun/rtn27/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1
    MODEL_PATH=/share/j_sun/rtn27/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_two_phase_hotpotqa_ep5_bsz48
    MODEL_PATH=/share/j_sun/rtn27/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz48_th-1_2phase_march8th_fixed
    DATABASE_PATH="na" # -> Not used for two phase
    TRAIN_SIZE=100
    DATASET_NAME="hotpotqa"
    DEBUG="--debug"
    TWO_PHASE="--two_phase"
    REWARD_FUNC="em_size"
    if [ "${NUM_DB_ROLLOUTS_SET}" = "0" ]; then NUM_DB_ROLLOUTS=4; fi
else
    echo "Invalid dataset: ${DATASET}"
    exit 1
fi


if [ ! -d "${MODEL_PATH}" ]; then
    echo "Model path directory does not exist: ${MODEL_PATH}"
    exit 1
fi

echo "Starting GRPO Training..."
bash /home/dg793/Multi-Hop-Reasoning/scripts/grpo_train.sh \
--model_path ${MODEL_PATH} \
--dataset_name ${DATASET_NAME} \
--database_path ${DATABASE_PATH} \
--train_size ${TRAIN_SIZE} \
${DEBUG} \
${TWO_PHASE} \
--reward_func ${REWARD_FUNC} \
--phase1_reward_type ${PHASE1_REWARD_TYPE} \
--phase1_prompt_type ${PHASE1_PROMPT_TYPE} \
--num_db_rollouts ${NUM_DB_ROLLOUTS} \
--phase1_db_weight_mode ${PHASE1_DB_WEIGHT_MODE} \
${LEARNING_RATE} \
${RETRIEVAL_THRESHOLD_ARG} \
${TOP_K_ARG}

echo "GRPO Training complete!"
