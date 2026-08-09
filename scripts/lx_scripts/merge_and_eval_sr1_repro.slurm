#!/bin/bash
#
# Merge a verl FSDP checkpoint to HF format, then dispatch Search-R1 evals
# (both n=1000 and full-dev) against the merged model via the shared
# eval_search_r1.slurm dispatcher.
#
# Usage:
#     sbatch scripts/lx_scripts/merge_and_eval_sr1_repro.slurm
#
#SBATCH -J sr1_merge_eval_qwen3-1.7b-repro
#SBATCH -o /home/lz586/icl/slurm_output/lmlm_multi_hop/sr1_merge_eval_qwen3-1.7b-repro_%j.out
#SBATCH -e /home/lz586/icl/slurm_output/lmlm_multi_hop/sr1_merge_eval_qwen3-1.7b-repro_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH --partition=kilian,default_partition

set -o pipefail

REPO=/share/j_sun/mx253_lmlm
CKPT_ROOT=/home/lz586/icl/Multi-Hop-Reasoning-pr44-full/rebuttal-search-r1/checkpoints/LMLM/qwen3-1.7b-searchR1-paper-repro-77beb7e/global_step_500/actor
MERGED_DIR="${CKPT_ROOT}_hf"
MODEL_TAG="qwen3-1.7b-sr-repro"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate mem

# ── Merge (skip if already done) ─────────────────────────────────────────────
if [[ -f "${MERGED_DIR}/config.json" && (-f "${MERGED_DIR}/model.safetensors" || -f "${MERGED_DIR}/pytorch_model.bin" || -n "$(ls ${MERGED_DIR}/*.safetensors 2>/dev/null)") ]]; then
    echo "[$(date +%T)] merged model already exists at ${MERGED_DIR}, skipping merge"
else
    echo "[$(date +%T)] merging FSDP shards from ${CKPT_ROOT} -> ${MERGED_DIR}"
    mkdir -p "${MERGED_DIR}"
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${CKPT_ROOT}" \
        --target_dir "${MERGED_DIR}"
    if [[ $? -ne 0 ]]; then
        echo "FATAL: merge failed" >&2
        exit 1
    fi
    echo "[$(date +%T)] merge complete"
    ls -la "${MERGED_DIR}"
fi

# ── Dispatch both evals via env-var overrides ────────────────────────────────
# eval_search_r1.slurm reads SR1_MODELS / NUM_SAMPLES / SAVE_VERSION from the
# env, so we don't have to touch MODEL_LST between the two dispatches.

echo "[$(date +%T)] dispatching n=1000 eval"
SR1_MODELS="${MODEL_TAG}:${MERGED_DIR}" \
    NUM_SAMPLES=1000 \
    SAVE_VERSION="_0808_sr1_n1000_${MODEL_TAG}_shuf" \
    bash "${REPO}/scripts/lx_scripts/eval_search_r1.slurm"

echo "[$(date +%T)] dispatching full-dev eval"
SR1_MODELS="${MODEL_TAG}:${MERGED_DIR}" \
    NUM_SAMPLES=20000 \
    SAVE_VERSION="_0808_sr1_full_dev_${MODEL_TAG}_shuf" \
    bash "${REPO}/scripts/lx_scripts/eval_search_r1.slurm"

echo "[$(date +%T)] all dispatched"
