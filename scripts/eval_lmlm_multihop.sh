MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_ep5_bsz48
DATASET=hotpotqa
SPLIT=dev
USE_INVERSES="" # or "--use-inverses"
NUM_SAMPLES=1000


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
        --use-inverses)
            USE_INVERSES="--use-inverses"
            shift
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
        START_IDX=0
    elif [ "${SPLIT}" = "train" ]; then
        echo "Using train set from GRPO"
        DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_train_start_idx_82347_nb_8100_database.json"
        START_IDX=82347
    else
        echo "Error: SPLIT must be either 'train' or 'dev', got '${SPLIT}'"
        exit 1
    fi
elif [ "${DATASET}" = "musique" ]; then
    if [ "${SPLIT}" = "dev" ]; then
        DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/musique_validation_42_1000_all_context_database.json"
        START_IDX=0
    elif [ "${SPLIT}" = "train" ]; then
        echo "There is no train database made for musique"
        exit 1
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
BATCH_SIZE=32
OUTPUT_DIR=./output
SETTING=distractor
SAVE_EVERY=64
SEED=42


if [[ "${MODEL_PATH}" == *"-nak"* ]]; then
    ADAPTIVE_K=""
else
    ADAPTIVE_K="--adaptive-k"
fi

# th-3
if [[ "${MODEL_PATH}" == *"-th-3"* ]]; then
    RETURN_TRIPLETS="--return-triplets"
else
    RETURN_TRIPLETS=""
fi


python scripts/eval_lmlm_multihop.py \
    --model-path ${MODEL_PATH} \
    --database-path ${DATABASE_PATH} \
    --method ${METHOD} \
    --max-tokens ${MAX_TOKENS} \
    --batch-size ${BATCH_SIZE} \
    --total-count ${NUM_SAMPLES} \
    --output-dir ${OUTPUT_DIR}/ \
    --split ${SPLIT} \
    --setting ${SETTING} \
    --dataset ${DATASET} \
    --seed ${SEED} \
    --save-every ${SAVE_EVERY}\
    --start-index ${START_IDX}\
    ${ADAPTIVE_K} \
    ${RETURN_TRIPLETS} \
    ${USE_INVERSES} \
    --eval \
    --resume
