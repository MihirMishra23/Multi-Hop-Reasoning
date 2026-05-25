#!/bin/bash
#
# Master script to run the complete Search-R1 training pipeline
#
# Usage:
#   bash run_full_pipeline.sh --model_path /path/to/model --model_size 1.7B
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH=""
MODEL_SIZE="1.7B"
SKIP_SETUP=false
SKIP_DATA=false
SKIP_INDEX=false

function usage() {
    echo "Usage: $0 --model_path <path> [options]"
    echo ""
    echo "Required:"
    echo "  --model_path PATH       Path to base model (Qwen3-1.7B or Qwen3-4B)"
    echo ""
    echo "Optional:"
    echo "  --model_size SIZE       Model size: 1.7B or 4B (default: 1.7B)"
    echo "  --skip_setup            Skip environment setup"
    echo "  --skip_data             Skip data preparation"
    echo "  --skip_index            Skip index building"
    echo ""
    echo "Example:"
    echo "  $0 --model_path /path/to/Qwen3-1.7B --model_size 1.7B"
    echo ""
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --model_size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --skip_setup)
            SKIP_SETUP=true
            shift
            ;;
        --skip_data)
            SKIP_DATA=true
            shift
            ;;
        --skip_index)
            SKIP_INDEX=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

if [ -z "${MODEL_PATH}" ]; then
    echo "❌ Error: --model_path is required"
    usage
fi

echo "════════════════════════════════════════════════════════════════"
echo "Search-R1 Full Pipeline"
echo "════════════════════════════════════════════════════════════════"
echo "Model:      ${MODEL_PATH}"
echo "Size:       ${MODEL_SIZE}"
echo "Skip setup: ${SKIP_SETUP}"
echo "Skip data:  ${SKIP_DATA}"
echo "Skip index: ${SKIP_INDEX}"
echo "════════════════════════════════════════════════════════════════"
echo ""

cd "${SCRIPT_DIR}"

# Step 1: Environment setup
if [ "${SKIP_SETUP}" = false ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Step 1/5: Setting up conda environments"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    bash setup_environment.sh
else
    echo "⏭️  Skipping environment setup"
fi

# Activate searchr1 environment
eval "$(conda shell.bash hook)"
conda activate searchr1

# Step 2: Prepare data
if [ "${SKIP_DATA}" = false ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Step 2/5: Preparing HotpotQA data"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ ! -f "data/hotpotqa_corpus.jsonl" ]; then
        echo "Preparing corpus..."
        python prepare_hotpotqa_corpus.py
    else
        echo "✓ Corpus already exists"
    fi

    if [ ! -f "data/hotpotqa_train.jsonl" ]; then
        echo "Preparing training data..."
        python prepare_hotpotqa_data.py
    else
        echo "✓ Training data already exists"
    fi
else
    echo "⏭️  Skipping data preparation"
fi

# Step 3: Build index
if [ "${SKIP_INDEX}" = false ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Step 3/5: Building retrieval index"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ ! -f "data/index/e5_Flat.index" ]; then
        echo "Building E5 index (this may take 10-20 minutes)..."
        bash build_index.sh
    else
        echo "✓ Index already exists"
    fi
else
    echo "⏭️  Skipping index building"
fi

# Step 4: Start retrieval server (in background)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4/5: Starting retrieval server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if server is already running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ Retrieval server already running"
else
    echo "Starting retrieval server in background..."
    echo "Logs will be written to: retrieval_server.log"

    # Start server in background
    nohup bash retrieval_launch.sh > retrieval_server.log 2>&1 &
    SERVER_PID=$!
    echo "Server PID: ${SERVER_PID}"

    # Wait for server to start
    echo "Waiting for server to start..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "✓ Server started successfully"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "❌ Error: Server failed to start"
            echo "Check retrieval_server.log for details"
            exit 1
        fi
        sleep 2
    done
fi

# Step 5: Start training
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 5/5: Starting Search-R1 training"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

bash train.sh \
    --model_path "${MODEL_PATH}" \
    --model_size "${MODEL_SIZE}"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✓ Pipeline complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Note: Retrieval server is still running in the background"
echo "To stop it, run: pkill -f retrieval_server.py"
echo ""
