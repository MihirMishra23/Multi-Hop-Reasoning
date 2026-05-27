#!/bin/bash
#SBATCH --job-name=judge-phase1
#SBATCH --partition=default_partition,aimi,dean
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/judge_phase1_%j.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/judge_phase1_%j.err

# Usage:
#   sbatch --export=MODEL=qwen3_1.7b_sft_n1000_prod menghan-scripts/judge_phase1.slurm
#
# MODEL is the file stem (no extension, no path) in KG_results_v2/inference/

set -e

if [[ -z "${MODEL}" ]]; then
    echo "ERROR: MODEL env var not set. Use: sbatch --export=MODEL=<stem> ..." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate lmlm_mx

cd /share/j_sun/mx253/Multi-Hop-Reasoning

CSV=/share/j_sun/mx253/Multi-Hop-Reasoning/KG_results_v2/inference/${MODEL}.csv
OUT=/share/j_sun/mx253/Multi-Hop-Reasoning/KG_results_v2/phase1_judge/phase1_${MODEL}.json

mkdir -p $(dirname "$OUT")

echo "========== Phase 1 LLM judge: ${MODEL} =========="
echo "CSV:    $CSV"
echo "OUT:    $OUT"
echo "Start:  $(date)"

python reward_hacking_evaluate/phase1/evaluate_phase1.py \
    --csv "$CSV" \
    --output "$OUT" \
    --concurrency 5

echo "Done:   $(date)"
