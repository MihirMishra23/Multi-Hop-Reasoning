#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SEARCH_R1_DIR="${SCRIPT_DIR}/Search-R1"

echo "════════════════════════════════════════════════════════════════"
echo "Search-R1 Environment Setup"
echo "════════════════════════════════════════════════════════════════"

# ── Search-R1 Main Environment ──────────────────────────────────────
echo ""
echo "┌─ Step 1: Creating searchr1 conda environment"
echo "└─────────────────────────────────────────────────────────────"

if conda env list | grep -q "^searchr1 "; then
    echo "⚠️  searchr1 environment already exists. Removing..."
    conda env remove -n searchr1 -y
fi

conda create -n searchr1 python=3.9 -y
echo "✓ Created searchr1 environment"

echo ""
echo "┌─ Step 2: Installing PyTorch and vLLM"
echo "└─────────────────────────────────────────────────────────────"

eval "$(conda shell.bash hook)"
conda activate searchr1

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
echo "✓ Installed PyTorch 2.4.0"

pip install vllm==0.6.3
echo "✓ Installed vLLM 0.6.3"

echo ""
echo "┌─ Step 3: Cloning and installing Search-R1"
echo "└─────────────────────────────────────────────────────────────"

if [ -d "${SEARCH_R1_DIR}" ]; then
    echo "⚠️  Search-R1 directory already exists. Removing..."
    rm -rf "${SEARCH_R1_DIR}"
fi

cd "${SCRIPT_DIR}"
git clone https://github.com/PeterGriffinJin/Search-R1.git
cd "${SEARCH_R1_DIR}"
pip install -e .
echo "✓ Installed Search-R1 (verl)"

echo ""
echo "┌─ Step 4: Installing flash-attn and wandb"
echo "└─────────────────────────────────────────────────────────────"

# Try to install flash-attn from prebuilt wheel
echo "Attempting to install flash-attn..."
FLASH_ATTN_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp39-cp39-linux_x86_64.whl"

if pip install "${FLASH_ATTN_WHEEL_URL}" 2>/dev/null; then
    echo "✓ Installed flash-attn from prebuilt wheel"
elif pip install flash-attn --no-build-isolation 2>/dev/null; then
    echo "✓ Installed flash-attn from source"
else
    echo "⚠️  Warning: flash-attn installation failed (optional, continuing...)"
    echo "    vLLM will work without flash-attn, but may be slower"
fi

pip install wandb
echo "✓ Installed wandb"

echo ""
echo "┌─ Step 5: Installing Multi-Hop-Reasoning"
echo "└─────────────────────────────────────────────────────────────"

cd "${REPO_ROOT}"
pip install -e .
echo "✓ Installed Multi-Hop-Reasoning"

# ── Retriever Environment (Optional) ────────────────────────────────
echo ""
echo "┌─ Step 6: Creating retriever conda environment"
echo "└─────────────────────────────────────────────────────────────"

if conda env list | grep -q "^retriever "; then
    echo "⚠️  retriever environment already exists. Removing..."
    conda env remove -n retriever -y
fi

conda create -n retriever python=3.10 -y
echo "✓ Created retriever environment"

conda activate retriever

echo ""
echo "┌─ Step 7: Installing retriever dependencies"
echo "└─────────────────────────────────────────────────────────────"

conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
echo "✓ Installed PyTorch with CUDA"

pip install "transformers>=4.51.0" "tokenizers>=0.21" accelerate datasets pyserini
echo "✓ Installed transformers >=4.51.0 (required for Qwen3), datasets, pyserini"

conda install -c pytorch -c nvidia faiss-gpu=1.8.0 -y
echo "✓ Installed faiss-gpu"

pip install uvicorn fastapi
echo "✓ Installed uvicorn, fastapi"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✓ Environment setup complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. conda activate searchr1"
echo "  2. Run prepare_hotpotqa_corpus.py to build the corpus"
echo "  3. Run build_index.sh to index the corpus"
echo "  4. Run train.sh to start training"
echo ""
