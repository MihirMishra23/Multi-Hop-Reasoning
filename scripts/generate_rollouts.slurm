#!/bin/bash
#SBATCH --mail-user=ryan.noonan@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=gen_rol
#SBATCH --nodes=1
#SBATCH -p jjs533
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

mkdir -p logs
source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate multi-hop-reasoning

cd "${SLURM_SUBMIT_DIR}"

DATABASE_PATH="/home/rtn27/Multi-Hop-Reasoning/src/database_creation/gemini/output_train_42_6000_date_12-10/database.json"
METADATA_PATH="/home/rtn27/Multi-Hop-Reasoning/src/database_creation/gemini/output_train_42_6000_date_12-10/metadata.json"
MODEL="gemini-3-pro-preview"
MAX_GENERATIONS=6                  # max bn db lookups
START_IDX=0                        # Starting index for dataset
NB_EXAMPLES=6000
MAX_RETRIES=3
MAX_CONCURRENT=150            # Maximum concurrent requests
HOTPOT_SETTING="distractor"        # distractor or fullwiki
SPLIT="train"            # train, validation, test
SEED=42
PROMPT_NAME="default" # Prompt name from lmlm_agent.json
DB_TOP_K=4                        # maximum number of results to retrieve
DB_THRESHOLD=0.6                   # threshold for database retrieval
ADAPTIVE_K=true                   # Only retrieve the first elements before the largest jump in cosine similarity in the first k+1 retrieved.
RETURN_TRIPLETS=true

echo "Running rollout generation with the following configuration:"
echo "  Database path: ${DATABASE_PATH}"
echo "  Metadata path: ${METADATA_PATH}"
echo "  Model: ${MODEL}"
echo "  Max generations: ${MAX_GENERATIONS}"
echo "  Start index: ${START_IDX}"
echo "  Number of examples: ${NB_EXAMPLES}"
echo "  Max concurrent requests: ${MAX_CONCURRENT}"
echo "  HotpotQA setting: ${HOTPOT_SETTING}"
echo "  Split: ${SPLIT}"
echo "  Seed: ${SEED}"
echo "  Prompt name: ${PROMPT_NAME}"
echo "  DB top-k: ${DB_TOP_K}"
echo "  DB threshold: ${DB_THRESHOLD}"
echo "  Adaptive-k: ${ADAPTIVE_K}"
echo ""

# Build the adaptive-k flag
if [ "${ADAPTIVE_K}" = true ]; then
  ADAPTIVE_K_FLAG="--adaptive-k"
else
  ADAPTIVE_K_FLAG="--no-adaptive-k"
fi


# Build the return-triplets flag
if [ "${RETURN_TRIPLETS}" = true ]; then
  RETURN_TRIPLETS_FLAG="--return-triplets"
else
  RETURN_TRIPLETS_FLAG="--no-return-triplets"
fi

python "scripts/generate_rollouts.py" \
  --database-path "${DATABASE_PATH}" \
  --metadata-path "${METADATA_PATH}" \
  --model "${MODEL}" \
  --max-generations ${MAX_GENERATIONS} \
  --start-idx ${START_IDX} \
  --nb-examples ${NB_EXAMPLES} \
  --max-retries ${MAX_RETRIES} \
  --max-concurrent ${MAX_CONCURRENT} \
  --hotpot-setting "${HOTPOT_SETTING}" \
  --split "${SPLIT}" \
  --seed ${SEED} \
  --prompt-name "${PROMPT_NAME}" \
  --db-top-k ${DB_TOP_K} \
  --db-threshold ${DB_THRESHOLD} \
  ${ADAPTIVE_K_FLAG} \
  ${RETURN_TRIPLETS_FLAG}
