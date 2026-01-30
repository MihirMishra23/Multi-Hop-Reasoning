#!/bin/bash

# Script to build knowledge database from multi-hop QA datasets

DB_PATH=""

# Get the directory where this script is located
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

# Configuration
DATASET="2wiki"              # hotpotqa or musique
HOTPOT_SETTING="distractor"      # distractor or fullwiki (only for hotpot_qa)
SPLIT="dev"               # train, validation, test
MODEL="gemini-2.5-flash"     # Gemini model name
NB_EXAMPLES=10   # Number of examples to process
SAMPLE_FROM="start"              # start or end
USE_CONTEXT=all            # 'all'' or 'golden'
PROMPT_NAME="default"            # Prompt name from prompts.json
SEED=42                          
MAX_CONCURRENT=150         
NB_PARTS_PER_PROMPT=5 # Only applies when using all contexts

echo "Running database extraction with the following configuration:"
echo "  Dataset: ${DATASET}"
if [ "${DATASET}" = "hotpotqa" ]; then
  echo "  HotpotQA setting: ${HOTPOT_SETTING}"
fi
echo "  Split: ${SPLIT}"
echo "  Model: ${MODEL}"
echo "  Number of examples: ${NB_EXAMPLES}"
echo "  Sample from: ${SAMPLE_FROM}"
echo "  Prompt name: ${PROMPT_NAME}"
echo "  Seed: ${SEED}"
echo "  Max concurrent requests: ${MAX_CONCURRENT}"
echo "  Using context: ${USE_CONTEXT}"
echo "  Nb parts per prompt: ${NB_PARTS_PER_PROMPT}"
echo ""

if [ -n "${DB_PATH}" ]; then
  echo "Output database bath set to: ${DB_PATH}"
  DB_PATH_FLAG="--database-path ${DB_PATH}"
else 
  DB_PATH_FLAG=""
fi


# Run the python script
python "${SCRIPT_DIR}/build_database.py" \
  --dataset "${DATASET}" \
  --hotpot-setting "${HOTPOT_SETTING}" \
  --split "${SPLIT}" \
  --model "${MODEL}" \
  --nb-examples ${NB_EXAMPLES} \
  --sample-from "${SAMPLE_FROM}" \
  --prompt-name "${PROMPT_NAME}" \
  --seed ${SEED} \
  --max-concurrent ${MAX_CONCURRENT} \
  --nb-parts-per-prompt ${NB_PARTS_PER_PROMPT} \
  --use-context ${USE_CONTEXT} \
  --count-tokens \
  ${DB_PATH_FLAG}
