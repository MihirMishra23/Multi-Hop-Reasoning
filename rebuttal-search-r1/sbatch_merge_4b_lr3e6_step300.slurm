#!/bin/bash
#
# One-shot merge: verl FSDP shards -> HF-format dir for qwen3-4b-searchR1-a2a-lr3e6-step300.

#SBATCH -J merge_4b_lr3e6_step300
#SBATCH -o /home/lz586/icl/slurm_output/searchr1_a2a/%x_%j.out
#SBATCH -e /home/lz586/icl/slurm_output/searchr1_a2a/%x_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH --partition=default_partition

set -euo pipefail

ENV_DIR="/home/lz586/.conda/envs/mem"
SRC="/share/j_sun/lmlm_multihop/searchr1_checkpoints/search-r1-hotpotqa/qwen3-4b-searchR1-a2a-lr3e6-n8-20260530_1023/global_step_300/actor"
DST="/home/lz586/icl/Multi-Hop-Reasoning/rebuttal-search-r1/merged_ckpts/qwen3-4b-searchR1-a2a-lr3e6-step300"

echo "[$(date)] === merge sbatch start ==="
echo "Node:   $(hostname)"
echo "SRC:    ${SRC}"
echo "DST:    ${DST}"

[ -d "${SRC}" ] || { echo "missing SRC"; exit 1; }
mkdir -p "${DST}"

export PATH="${ENV_DIR}/bin:${PATH}"
python3 --version

echo "[$(date)] launching verl.model_merger merge ..."
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${SRC}" \
    --target_dir "${DST}" \
    --use_cpu_initialization

echo "[$(date)] === merge done ==="
ls -lh "${DST}/"
du -sh "${DST}/"
