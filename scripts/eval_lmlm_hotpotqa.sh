MODEL_PATH=/home/rtn27/LMLM_develop/training/qwen3-1.7b/checkpoints/_full_ep10_bsz32_new_qa
DATABASE_PATH=/home/rtn27/LMLM/build-database/triplets/hotpotqa_1k_42_dev_triplets.json
METHOD=lmlm
MAX_TOKENS=512
BATCH_SIZE=4
NUMB_BATCHES=-1
OUTPUT_DIR=/home/rtn27/Multi-Hop-Reasoning/qwen3-1.7b
SPLIT=dev
SETTING=distractor
DATASET=hotpotqa


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
    --num-batches ${NUMB_BATCHES}