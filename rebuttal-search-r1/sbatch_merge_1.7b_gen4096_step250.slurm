#!/bin/bash
#
# Merge verl FSDP shards -> HF dir for 1.7B gen=4096 run at step 250 (val-peak ckpt).

#SBATCH -J merge_1.7b_gen4096_step250
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
SRC="/share/j_sun/lmlm_multihop/searchr1_checkpoints/search-r1-hotpotqa/qwen3-1.7b-searchR1-a2a-lr3e6-n8-icl3hop-gen4096-noth-8gpu-W4-20260602_1532/global_step_250/actor"
DST="/home/lz586/icl/Multi-Hop-Reasoning/rebuttal-search-r1/merged_ckpts/qwen3-1.7b-icl3hop-gen4096-step250"

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
