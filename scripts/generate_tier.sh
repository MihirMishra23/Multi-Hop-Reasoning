#
# Setup instructions — generate_tier.sh
#
# Prereqs:
# - Activate your environment and install repo deps:
#   pip install -e .
# - If you want a plot, install matplotlib:
#   pip install matplotlib
#
# Usage:
# - Set defaults in this file, or override via CLI args.
# - Run:
#   bash scripts/generate_tier.sh
#
# Example override:
#   bash scripts/generate_tier.sh \
#     --model_path /path/to/checkpoint \
#     --dataset 2wiki \
#     --split dev \
#     --num_samples 200 \
#     --num_rollouts 8 \
#     --answer_threshold 0.7 \
#     --save_version v2 \
#     --plot_path ./output/tiers/2wiki_score_dist_v2.png
#
# Output:
# - JSON with id/questions/answers/traces + score (0..num_rollouts)
# - Optional score distribution plot at --plot_path
#
# Compatibility with eval:
# - Uses get_dataset with seed; subset defined by start-index + total-count
#
MODEL_PATH=/share/j_sun/rtn27/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k
DATASET=hotpotqa
# train_all: full 7k GRPO training set (seed=42 shuffled, indices 83347..90347)
SPLIT=train_all
NUM_SAMPLES=7000
SAVE_VERSION="v1"
NUM_ROLLOUTS=8
ANSWER_THRESHOLD=0.6
PLOT_PATH=./output/tiers/hotpotqa_train_score_dist.png


# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --num_rollouts)
            NUM_ROLLOUTS="$2"
            shift 2
            ;;
        --answer_threshold)
            ANSWER_THRESHOLD="$2"
            shift 2
            ;;
        --save_version)
            SAVE_VERSION="$2"
            shift 2
            ;;
        --plot_path)
            PLOT_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ "${DATASET}" = "hotpotqa" ]; then
    if [ "${SPLIT}" = "dev" ]; then
        DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_validation_42_1000_all_context_database.json"
        DEFAULT_NUM_SAMPLES=1000
        START_IDX=0
    elif [ "${SPLIT}" = "debug_dev" ]; then
        DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/generated_database_validation_42_1000.json"
        DEFAULT_NUM_SAMPLES=1000
        START_IDX=0
        SPLIT="dev"
    elif [ "${SPLIT}" = "train_val100" ]; then
        echo "Using eval set from GRPO"
        DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_train_start_idx_82347_nb_8100_database.json"
        DEFAULT_NUM_SAMPLES=100
        START_IDX=90347
        SPLIT="train"
    elif [ "${SPLIT}" = "train_val1k" ]; then
        echo "Using train set from GRPO"
        DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_train_start_idx_82347_nb_8100_database.json"
        DEFAULT_NUM_SAMPLES=1000
        START_IDX=82347
        SPLIT="train"
    elif [ "${SPLIT}" = "train_train1k" ]; then
        echo "Using train set from GRPO"
        DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_train_start_idx_82347_nb_8100_database.json"
        DEFAULT_NUM_SAMPLES=1000
        START_IDX=89347
        SPLIT="train"
    elif [ "${SPLIT}" = "train_all" ]; then
        # Full 7k GRPO training set.
        # grpo_train.py loads sub_split="train", limit=7000, seed=42 which selects
        # range(n - eval_size - train_size, n - eval_size) = range(83347, 90347)
        # of the seed=42-shuffled hotpotqa train split (n=90447).
        echo "Using full 7k GRPO training set (seed=42 indices 83347..90347)"
        DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_train_start_idx_82347_nb_8100_database.json"
        DEFAULT_NUM_SAMPLES=7000
        START_IDX=83347
        SPLIT="train"
    else
        echo "Error: SPLIT must be either 'train' or 'dev', got '${SPLIT}'"
        exit 1
    fi
elif [ "${DATASET}" = "musique" ]; then
    if [ "${SPLIT}" = "dev" ]; then
        DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/musique_validation_42_1000_all_context_database.json"
        DEFAULT_NUM_SAMPLES=1000
        START_IDX=0
    elif [ "${SPLIT}" = "train" ]; then
        echo "There is no train database made for musique"
        exit 1
    else
        echo "Error: SPLIT must be either 'train' or 'dev', got '${SPLIT}'"
        exit 1
    fi
elif [ "${DATASET}" = "two_wiki" ] || [ "${DATASET}" = "2wiki" ]; then
    if [ "${SPLIT}" = "dev" ]; then
        DATABASE_PATH="/share/j_sun/as2637/database/2wiki_db.json"
        DEFAULT_NUM_SAMPLES=1000
        START_IDX=0
    else
        echo "Error: SPLIT must be either 'train' or 'dev', got '${SPLIT}'"
        exit 1
    fi
else
    echo "Error: DATASET must be one of 'hotpotqa', 'musique', or 'two_wiki', got '${DATASET}'"
    exit 1
fi

if [ "${NUM_SAMPLES}" -gt "${DEFAULT_NUM_SAMPLES}" ]; then
    NUM_SAMPLES="${DEFAULT_NUM_SAMPLES}"
fi

METHOD=lmlm
MAX_TOKENS=1024
BATCH_SIZE=32
OUTPUT_DIR=./output
SETTING=distractor
SEED="${SEED:-42}"

python scripts/generate_tier.py \
    --model-path ${MODEL_PATH} \
    --database-path ${DATABASE_PATH} \
    --method ${METHOD} \
    --max-tokens ${MAX_TOKENS} \
    --batch-size ${BATCH_SIZE} \
    --total-count ${NUM_SAMPLES} \
    --output-dir ${OUTPUT_DIR}/ \
    --save-version ${SAVE_VERSION} \
    --split ${SPLIT} \
    --setting ${SETTING} \
    --dataset ${DATASET} \
    --seed ${SEED} \
    --start-index ${START_IDX} \
    --num-rollouts ${NUM_ROLLOUTS} \
    --answer-threshold ${ANSWER_THRESHOLD} \
    --plot-path ${PLOT_PATH}
