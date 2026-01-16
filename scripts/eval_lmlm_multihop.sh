MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_ep5_bsz48
DATASET=hotpotqa

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
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Set DATABASE_PATH based on dataset
if [ "${DATASET}" = "hotpotqa" ]; then
    DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_validation_42_1000_all_context_database.json"
elif [ "${DATASET}" = "musique" ]; then
    DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/musique_validation_42_1000_all_context_database.json"
else
    echo "Error: DATASET must be either 'hotpotqa' or 'musique', got '${DATASET}'"
    exit 1
fi

METHOD=lmlm
MAX_TOKENS=1024
NUMB_BATCHES=1
BATCH_SIZE=1000
OUTPUT_DIR=./output
SPLIT=dev
SETTING=distractor

SEED=42
ADAPTIVE_K=true



python scripts/eval_lmlm_multihop.py \
    --model-path ${MODEL_PATH} \
    --database-path ${DATABASE_PATH} \
    --method ${METHOD} \
    --max-tokens ${MAX_TOKENS} \
    --batch-size ${BATCH_SIZE} \
    --output-dir ${OUTPUT_DIR}/ \
    --split ${SPLIT} \
    --setting ${SETTING} \
    --dataset ${DATASET} \
    --num-batches ${NUMB_BATCHES} \
    --seed ${SEED} \
    --adaptive-k ${ADAPTIVE_K} \
    --eval


# PRED_FILE=/home/lz586/icl/Multi-Hop-Reasoning/output/Qwen3-1.7B-SFT_ep5_bsz48.json
# python scripts/evaluate.py \
#     --preds ${PRED_FILE} \
#     --split ${SPLIT} \
#     --seed ${SEED} \
#     --batch-size ${BATCH_SIZE} \
#     --outdir ${OUTPUT_DIR}/results
