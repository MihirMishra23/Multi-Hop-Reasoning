MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_ep5_bsz48
# BUG: why is the hotpotqa database?
DATABASE_PATH=/home/rtn27/LMLM/build-database/triplets/hotpotqa_1k_42_dev_triplets.json
METHOD=lmlm
MAX_TOKENS=1024
BATCH_SIZE=100
NUMB_BATCHES=10
OUTPUT_DIR=./results
SPLIT=dev
SETTING=distractor
DATASET=musique
SEED=42

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

python scripts/run_agent.py \
    --model-path ${MODEL_PATH} \
    --database-path ${DATABASE_PATH} \
    --method ${METHOD} \
    --max-tokens ${MAX_TOKENS} \
    --batch-size ${BATCH_SIZE} \
    --output-dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --setting ${SETTING} \
    --dataset ${DATASET} \
    --num-batches ${NUMB_BATCHES} \
    --seed ${SEED}