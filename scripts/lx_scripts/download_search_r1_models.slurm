#!/bin/bash
#
# Download the Search-R1 v0.1 checkpoints into the shared HF cache.
#
# Why a batch job and not just `hf download` on the login node: the login node
# kills long-running / high-bandwidth processes, so downloads there die partway
# through (observed twice, at ~2.8G of 6G and ~1.8G of 15G). Compute nodes have
# outbound network and no such watchdog.
#
# CPU-only on purpose — this is pure I/O, requesting a GPU would waste one.
#
# Usage:
#     sbatch scripts/lx_scripts/download_search_r1_models.slurm
#
#SBATCH -J sr1_download
#SBATCH -o /home/lz586/icl/slurm_output/lmlm_multi_hop/sr1_download_%j.out
#SBATCH -e /home/lz586/icl/slurm_output/lmlm_multi_hop/sr1_download_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 08:00:00
#SBATCH --partition=kilian,default_partition
#SBATCH --requeue

source activate mem

MODELS=(
    "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo"
    "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo"
)

echo "[$(date +%T)] node=$(hostname)"

for M in "${MODELS[@]}"; do
    echo "[$(date +%T)] downloading ${M}"
    # Retry a few times: partial files resume, so a dropped connection costs
    # only the current chunk rather than the whole model.
    for attempt in 1 2 3; do
        if hf download "${M}"; then
            echo "[$(date +%T)] OK ${M}"
            break
        fi
        echo "[$(date +%T)] attempt ${attempt} failed for ${M}, retrying"
        sleep 30
    done
done

echo "[$(date +%T)] verifying (no .incomplete left)"
for M in "${MODELS[@]}"; do
    DIR="${HOME}/.cache/huggingface/hub/models--${M//\//--}"
    N=$(find "${DIR}" -name '*.incomplete' 2>/dev/null | wc -l)
    SZ=$(du -sh "${DIR}" 2>/dev/null | cut -f1)
    echo "  ${M}: size=${SZ} incomplete=${N}"
done

echo "[$(date +%T)] done"
