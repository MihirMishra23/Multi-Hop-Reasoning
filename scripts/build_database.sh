#!/bin/bash

# Script to build knowledge database from multi-hop QA datasets

# Get the directory where this script is located
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

# Configuration
DATASET="hotpot_qa"              # hotpot_qa or musique
HOTPOT_SETTING="distractor"      # distractor or fullwiki (only for hotpot_qa)
SPLIT="validation"               # train, validation, test
MODEL="gemini-3-pro-preview"     # Gemini model name
NB_EXAMPLES=100      # Number of examples to process
SAMPLE_FROM="start"              # start or end
USE_ONLY_GOLDEN=false            # true or false
PROMPT_NAME="default"            # Prompt name from prompts.json
SEED=42                          
MAX_CONCURRENT=150         
NB_PARTS_PER_PROMPT=5 # Only applies when using all contexts

echo "Running database extraction with the following configuration:"
echo "  Dataset: ${DATASET}"
if [ "${DATASET}" = "hotpot_qa" ]; then
  echo "  HotpotQA setting: ${HOTPOT_SETTING}"
fi
echo "  Split: ${SPLIT}"
echo "  Model: ${MODEL}"
echo "  Number of examples: ${NB_EXAMPLES}"
echo "  Sample from: ${SAMPLE_FROM}"
echo "  Prompt name: ${PROMPT_NAME}"
echo "  Seed: ${SEED}"
echo "  Max concurrent requests: ${MAX_CONCURRENT}"
echo "  Use only golden contexts: ${USE_ONLY_GOLDEN}"
echo ""

# Build the use-only-golden flag
if [ "${USE_ONLY_GOLDEN}" = true ]; then
  USE_ONLY_GOLDEN_FLAG="--use-only-golden"
else
  USE_ONLY_GOLDEN_FLAG="--no-use-only-golden"
fi


# Run the python script
python "${SCRIPT_DIR}/extract_triplets.py" \
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
  ${USE_ONLY_GOLDEN_FLAG}
