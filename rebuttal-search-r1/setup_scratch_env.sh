#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH_DIR="/scratch/rtn27"

if [ ! -d "${SCRATCH_DIR}" ]; then
    echo "❌ Error: /scratch/rtn27 does not exist"
    exit 1
fi

ENVS_DIR="${SCRATCH_DIR}/envs"
mkdir -p "${ENVS_DIR}"

echo "════════════════════════════════════════════════════════════════"
echo "Building Conda Environments on /scratch"
echo "════════════════════════════════════════════════════════════════"
echo "Location: ${ENVS_DIR}"
echo ""

# ── Environment 1: searchr1 (main training env) ─────────────────────

echo "📦 Creating searchr1 environment on /scratch..."
conda create -p "${ENVS_DIR}/searchr1" python=3.10 -y

echo "📦 Installing PyTorch 2.4.0 with CUDA 12.1..."
conda run -p "${ENVS_DIR}/searchr1" pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

echo "📦 Installing vLLM (latest for Qwen3 support)..."
conda run -p "${ENVS_DIR}/searchr1" pip install vllm

echo "📦 Installing transformers >=4.51.0 (required for Qwen3)..."
conda run -p "${ENVS_DIR}/searchr1" pip install "transformers>=4.51.0" "tokenizers>=0.21" accelerate

echo "📦 Installing flash-attn (optional, may take time)..."
FLASH_ATTN_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.4cxx11abiFALSE-cp39-cp39-linux_x86_64.whl"
if conda run -p "${ENVS_DIR}/searchr1" pip install "${FLASH_ATTN_WHEEL_URL}" 2>/dev/null; then
    echo "✓ Installed flash-attn from prebuilt wheel"
elif conda run -p "${ENVS_DIR}/searchr1" pip install flash-attn --no-build-isolation 2>/dev/null; then
    echo "✓ Installed flash-attn from source"
else
    echo "⚠️  Warning: flash-attn installation failed (optional, continuing...)"
fi

echo "📦 Cloning Search-R1 repository..."
if [ ! -d "${SCRIPT_DIR}/Search-R1" ]; then
    cd "${SCRIPT_DIR}"
    git clone https://github.com/PeterGriffinJin/Search-R1.git
fi

echo "📦 Installing Search-R1 (verl)..."
cd "${SCRIPT_DIR}/Search-R1"
conda run -p "${ENVS_DIR}/searchr1" pip install -e .

echo "📦 Installing additional dependencies..."
conda run -p "${ENVS_DIR}/searchr1" pip install \
    datasets \
    pyarrow \
    fastparquet \
    wandb \
    hydra-core \
    omegaconf \
    ray

echo "✓ searchr1 environment ready at ${ENVS_DIR}/searchr1"
echo ""

# ── Environment 2: retriever (retrieval server env) ────────────────

echo "📦 Creating retriever environment on /scratch..."
conda create -p "${ENVS_DIR}/retriever" python=3.10 -y

echo "📦 Installing PyTorch for retriever..."
conda run -p "${ENVS_DIR}/retriever" pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "📦 Installing retrieval dependencies..."
conda run -p "${ENVS_DIR}/retriever" pip install \
    sentence-transformers \
    faiss-gpu \
    fastapi \
    uvicorn

echo "✓ retriever environment ready at ${ENVS_DIR}/retriever"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "✓ Setup complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Environments created at:"
echo "  - ${ENVS_DIR}/searchr1"
echo "  - ${ENVS_DIR}/retriever"
echo ""
echo "To activate:"
echo "  conda activate ${ENVS_DIR}/searchr1"
echo "  conda activate ${ENVS_DIR}/retriever"
echo ""
echo "⚠️  IMPORTANT: You need to update train.sh and retrieval_launch.sh"
echo "   to use these new environment paths. Run:"
echo "   bash update_scripts_for_scratch.sh"
echo ""
