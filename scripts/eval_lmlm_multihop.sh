MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_ep5_bsz48
DATASET=hotpotqa
SPLIT=dev

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
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ "${DATASET}" = "hotpotqa" ]; then
    if [ "${SPLIT}" = "dev" ]; then
        DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_validation_42_1000_all_context_database.json"
    elif [ "${SPLIT}" = "train" ]; then
        echo "Using train set from GRPO"
        DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_train_start_idx_82347_nb_8100_database.json"
    else
        echo "Error: SPLIT must be either 'train' or 'dev', got '${SPLIT}'"
        exit 1
    fi
elif [ "${DATASET}" = "musique" ]; then
    if [ "${SPLIT}" = "dev" ]; then
        DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/musique_validation_42_1000_all_context_database.json"
    elif [ "${SPLIT}" = "train" ]; then
        echo "There is no train database made for musique"
    else
        echo "Error: SPLIT must be either 'train' or 'dev', got '${SPLIT}'"
        exit 1
    fi
else
    echo "Error: DATASET must be either 'hotpotqa' or 'musique', got '${DATASET}'"
    exit 1
fi

METHOD=lmlm
MAX_TOKENS=1024
TOTAL_COUNT=30
BATCH_SIZE=16
OUTPUT_DIR=./output
SETTING=distractor
SAVE_EVERY=64
SEED=42
ADAPTIVE_K=true
START_IDX=0


python scripts/eval_lmlm_multihop.py \
    --model-path ${MODEL_PATH} \
    --database-path ${DATABASE_PATH} \
    --method ${METHOD} \
    --max-tokens ${MAX_TOKENS} \
    --batch-size ${BATCH_SIZE} \
    --total-count ${TOTAL_COUNT} \
    --output-dir ${OUTPUT_DIR}/ \
    --split ${SPLIT} \
    --setting ${SETTING} \
    --dataset ${DATASET} \
    --seed ${SEED} \
    --adaptive-k ${ADAPTIVE_K} \
    --save-every ${SAVE_EVERY}\
    --start-index ${START_IDX}\
    --eval \
    --resume
