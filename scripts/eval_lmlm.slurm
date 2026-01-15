#!/bin/bash
#SBATCH --job-name=1.7b-t-eval
#SBATCH --nodes=1
#SBATCH -p jjs533
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --constraint='6000ada'
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err


set -euo pipefail

# Run from the directory where you submitted the job
cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs
source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate multi-hop-reasoning

echo "Running eval..."

DATASET='hotpotqa'

# Set DATABASE_PATH based on dataset
if [ "${DATASET}" = "hotpotqa" ]; then
    DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_validation_42_1000_all_context_database.json"
elif [ "${DATASET}" = "musique" ]; then
    DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/musique_validation_42_1000_all_context_database.json"
else
    echo "Error: DATASET must be either 'hotpotqa' or 'musique', got '${DATASET}'"
    exit 1
fi


MODEL_PATH=/share/j_sun/rtn27/checkpoints/qwen3-1.7B_sft_v1.3_5743/sft_ep5_bsz32_new_qa_26-01-06_04-44
DATABASE_PATH=/home/rtn27/Multi-Hop-Reasoning/src/database_creation/gemini/hotpotqa_validation_all_context_42_start_idx_0_1000_date_12-29/database.json
OUTPUT_DIR="/share/j_sun/rtn27/preds/qwen3-1.7B_sft_v1.3_5743_triplets/"
METHOD=lmlm
MAX_TOKENS=2048
NUMB_BATCHES=1
BATCH_SIZE=10
SPLIT=dev
SETTING=distractor
SEED=42
ADAPTIVE_K=false
TOP_K=10

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
    --seed ${SEED} \
    --adaptive-k ${ADAPTIVE_K} \
    --top-k ${TOP_K}

echo "Successfully saved predictions to ${OUTPUT_DIR}"
