#!/bin/bash
#
# Search-R1 setup on a fresh B200 box for Qwen3-1.7B.
#
# Two install paths — pick one by setting STACK before running:
#
#   STACK=modern  (default) torch 2.8 + vllm 0.10 + transformers 4.57 + verl from
#                 Search-R1 fork installed with --no-deps. Mirrors what already
#                 worked for Qwen3 generation in the `mem` env on the CS NFS box.
#                 Caveat: Search-R1's `verl.third_party.vllm.vllm_v_0_6_3`
#                 shim is incompatible with vllm 0.10; the train loop monkey-
#                 patches vLLM via Search-R1 wrappers. If training errors on
#                 vllm import, fall back to STACK=pinned.
#
#   STACK=pinned  Search-R1 README pin: python 3.10 + torch 2.4 cu121 +
#                 vllm 0.6.3 + transformers 4.51 + Search-R1 deps. Closer to
#                 paper-reported config, but vllm 0.6.3 does NOT support Qwen3
#                 natively — needs a manual qwen3.py patch under
#                 site-packages/vllm/model_executor/models/ (not checked in).
#
# Usage:
#   bash setup_b200.sh                    # modern stack, env name searchr1-b200
#   STACK=pinned ENV_NAME=sr1 bash setup_b200.sh
#   SKIP_DATA=1 bash setup_b200.sh        # don't rsync data from CS NFS

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Either ENV_NAME (conda env in default envs dir) or ENV_PATH (absolute path,
# useful for /scratch on B200 to avoid slow NFS writes).
ENV_NAME="${ENV_NAME:-searchr1-b200}"
ENV_PATH="${ENV_PATH:-}"
STACK="${STACK:-modern}"
SKIP_DATA="${SKIP_DATA:-0}"
SKIP_SANITY="${SKIP_SANITY:-0}"

if [ -n "${ENV_PATH}" ]; then
    ENV_SPEC=( -p "${ENV_PATH}" )
    ENV_ACTIVATE="${ENV_PATH}"
else
    ENV_SPEC=( -n "${ENV_NAME}" )
    ENV_ACTIVATE="${ENV_NAME}"
fi

# Host that holds the prepared corpus/parquets/index (~250MB total).
# Override if not pulling from CS NFS.
DATA_SRC_HOST="${DATA_SRC_HOST:-lz586@<cs-jump-host>}"
DATA_SRC_PATH="${DATA_SRC_PATH:-/home/lz586/icl/Multi-Hop-Reasoning/rebuttal-search-r1/data}"

echo "════════════════════════════════════════════════════════════════"
echo "Search-R1 B200 setup"
echo "  ENV      : ${ENV_ACTIVATE}"
echo "  STACK    : ${STACK}"
echo "  SCRIPT   : ${SCRIPT_DIR}"
echo "════════════════════════════════════════════════════════════════"

# ── 1. Create env ────────────────────────────────────────────────────
eval "$(conda shell.bash hook)"

env_exists() {
    if [ -n "${ENV_PATH}" ]; then
        [ -d "${ENV_PATH}/bin" ]
    else
        conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"
    fi
}

if env_exists; then
    echo "⚠️  Env ${ENV_ACTIVATE} exists, reusing."
else
    case "${STACK}" in
        modern)  PYVER=3.11 ;;
        pinned)  PYVER=3.10 ;;
        *) echo "Unknown STACK=${STACK}"; exit 1 ;;
    esac
    conda create "${ENV_SPEC[@]}" "python=${PYVER}" -y
fi
conda activate "${ENV_ACTIVATE}"

# ── 2. Install Python deps ──────────────────────────────────────────
case "${STACK}" in
modern)
    # Matches what already runs Qwen3 in `mem`.
    pip install --upgrade pip
    pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    pip install \
        "vllm==0.10.2" \
        "transformers==4.57.6" \
        "tokenizers>=0.21" \
        accelerate \
        datasets pyarrow fastparquet \
        ray==2.53.0 hydra-core omegaconf \
        wandb \
        fastapi uvicorn sentence-transformers
    # faiss-gpu is unmaintained on PyPI for py3.11. Prefer the official CUDA 12
    # wheel (faiss-gpu-cu12); fall back to CPU if even that fails.
    pip install faiss-gpu-cu12 || pip install faiss-cpu
    ;;
pinned)
    pip install --upgrade pip
    pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
    pip install \
        "vllm==0.6.3" \
        "transformers>=4.51.0" \
        "tokenizers>=0.21" \
        accelerate \
        datasets pyarrow fastparquet \
        ray hydra-core omegaconf \
        wandb \
        fastapi uvicorn sentence-transformers
    pip install faiss-gpu-cu12 || pip install faiss-cpu

    # Optional: prebuilt flash-attn for the pinned combo
    FLASH_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    pip install "${FLASH_WHL}" || pip install flash-attn --no-build-isolation || \
        echo "⚠️  flash-attn install skipped (optional)."
    ;;
esac

# ── 3. Clone Search-R1 ──────────────────────────────────────────────
SR1_DIR="${SCRIPT_DIR}/Search-R1"
if [ ! -d "${SR1_DIR}" ]; then
    git clone https://github.com/PeterGriffinJin/Search-R1.git "${SR1_DIR}"
fi

# Install Search-R1's vendored verl. For modern stack we --no-deps to avoid
# overwriting torch/vllm/transformers; for pinned stack let it resolve.
if [ "${STACK}" = "modern" ]; then
    pip install -e "${SR1_DIR}" --no-deps
else
    pip install -e "${SR1_DIR}"
fi

# Install the parent Multi-Hop-Reasoning package so data prep scripts can
# `from src.data import get_dataset` (corpus/data prep only — not needed on B200
# if you already have the parquets+index from rsync).
pip install -e "$(cd "${SCRIPT_DIR}/.." && pwd)" --no-deps || true

# ── 4. Pull pre-built data from the CS box ──────────────────────────
if [ "${SKIP_DATA}" != "1" ]; then
    mkdir -p "${SCRIPT_DIR}/data"
    if [ ! -f "${SCRIPT_DIR}/data/train.parquet" ]; then
        echo ""
        echo "Rsyncing data (~250MB) from ${DATA_SRC_HOST}:${DATA_SRC_PATH}"
        echo "  (set DATA_SRC_HOST=... or SKIP_DATA=1 to skip)"
        rsync -avh --progress \
            "${DATA_SRC_HOST}:${DATA_SRC_PATH}/" \
            "${SCRIPT_DIR}/data/" || {
                echo "⚠️  rsync failed. Either fix DATA_SRC_HOST or run"
                echo "    prepare_hotpotqa_corpus.py + prepare_hotpotqa_data.py + build_index.sh"
                echo "    on this box directly."
            }
    else
        echo "✓ data/ already populated."
    fi
fi

# ── 5. Sanity check: can Qwen3 generate? ────────────────────────────
if [ "${SKIP_SANITY}" = "1" ]; then
    echo "⏭️  Skipping Qwen3 sanity check (SKIP_SANITY=1)"
    exit 0
fi
echo ""
echo "── Qwen3-1.7B sanity check (small generate via vLLM) ────────────"
python - <<'PY'
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen3-1.7B", trust_remote_code=True,
          gpu_memory_utilization=0.25, enforce_eager=True)
out = llm.generate(
    ["<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"],
    SamplingParams(temperature=0.7, top_p=0.9, max_tokens=32),
)
txt = out[0].outputs[0].text
print("GEN:", repr(txt))
assert txt.strip() and not all(c == "!" for c in txt.strip()), \
    "Qwen3 generated garbage — vllm needs the Qwen3 patch or a version bump"
print("✓ Qwen3 generation looks healthy")
PY

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✓ Setup complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  # Terminal 1 — retrieval server"
echo "  CONDA_ENV=${ENV_ACTIVATE} bash retrieval_launch.sh"
echo ""
echo "  # Terminal 2 — training (recommend 4 B200s for Qwen3-1.7B)"
echo "  CONDA_ENV=${ENV_ACTIVATE} bash train.sh --model_size 1.7B --num_gpus 4"
echo ""
