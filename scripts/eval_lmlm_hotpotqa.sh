MODEL_PATH=/home/rtn27/Multi-Hop-Reasoning/scripts/training/Qwen3-1.7B/nov29/checkpoints/_full_ep5_bsz32_new_qa
DATABASE_PATH=/home/rtn27/Multi-Hop-Reasoning/src/database-creation/build-database-gemini/generated_database_validation_42_1000.json
METHOD=lmlm
MAX_TOKENS=512
BATCH_SIZE=1000
NUMB_BATCHES=1
OUTPUT_DIR=/home/rtn27/Multi-Hop-Reasoning/qwen3-1.7b
SPLIT=dev
SETTING=distractor
DATASET=hotpotqa
SEED=42


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