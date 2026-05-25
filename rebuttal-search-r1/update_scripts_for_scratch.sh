#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH_SEARCHR1="/scratch/rtn27/envs/searchr1"
SCRATCH_RETRIEVER="/scratch/rtn27/envs/retriever"

echo "════════════════════════════════════════════════════════════════"
echo "Updating Scripts to Use /scratch Conda Environments"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check if environments exist
if [ ! -d "${SCRATCH_SEARCHR1}" ]; then
    echo "❌ Error: searchr1 environment not found at ${SCRATCH_SEARCHR1}"
    echo "   Please run setup_scratch_env.sh first"
    exit 1
fi

if [ ! -d "${SCRATCH_RETRIEVER}" ]; then
    echo "❌ Error: retriever environment not found at ${SCRATCH_RETRIEVER}"
    echo "   Please run setup_scratch_env.sh first"
    exit 1
fi

# Update train.sh
echo "📝 Updating train.sh..."
sed -i 's|conda activate searchr1|conda activate '"${SCRATCH_SEARCHR1}"'|g' "${SCRIPT_DIR}/train.sh"
echo "✓ train.sh updated"

# Update retrieval_launch.sh
echo "📝 Updating retrieval_launch.sh..."
sed -i 's|conda activate retriever|conda activate '"${SCRATCH_RETRIEVER}"'|g' "${SCRIPT_DIR}/retrieval_launch.sh"
echo "✓ retrieval_launch.sh updated"

# Update build_index.sh
echo "📝 Updating build_index.sh..."
sed -i 's|conda activate retriever|conda activate '"${SCRATCH_RETRIEVER}"'|g' "${SCRIPT_DIR}/build_index.sh"
echo "✓ build_index.sh updated"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✓ All scripts updated!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Updated paths:"
echo "  train.sh:            ${SCRATCH_SEARCHR1}"
echo "  retrieval_launch.sh: ${SCRATCH_RETRIEVER}"
echo "  build_index.sh:      ${SCRATCH_RETRIEVER}"
echo ""
echo "You can now run training with:"
echo "  bash train.sh --model_size 1.7B"
echo ""
