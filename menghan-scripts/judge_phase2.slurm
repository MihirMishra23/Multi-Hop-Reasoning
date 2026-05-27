#!/bin/bash
#SBATCH --job-name=judge-phase2
#SBATCH --partition=default_partition,aimi,dean
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/judge_phase2_%j.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/judge_phase2_%j.err

# Usage (ORIGINAL CSV):
#   sbatch --export=MODEL=qwen3_1.7b_sft_n1000_prod menghan-scripts/judge_phase2.slurm
#
# Usage (FILTERED CSV, after running posthoc_filter_kb.py):
#   sbatch --export=MODEL=qwen3_1.7b_sft_n1000_prod,SUFFIX=_filtered menghan-scripts/judge_phase2.slurm

set -e

if [[ -z "${MODEL}" ]]; then
    echo "ERROR: MODEL env var not set. Use: sbatch --export=MODEL=<stem>[,SUFFIX=_filtered] ..." >&2
    exit 1
fi

SUFFIX="${SUFFIX:-}"

eval "$(conda shell.bash hook)"
conda activate lmlm_mx

cd /share/j_sun/mx253/Multi-Hop-Reasoning

CSV=/share/j_sun/mx253/Multi-Hop-Reasoning/KG_results_v2/inference/${MODEL}${SUFFIX}.csv
OUT=/share/j_sun/mx253/Multi-Hop-Reasoning/KG_results_v2/phase2_judge/phase2_${MODEL}${SUFFIX}.json

mkdir -p $(dirname "$OUT")

echo "========== Phase 2 LLM judge: ${MODEL}${SUFFIX} =========="
echo "CSV:    $CSV"
echo "OUT:    $OUT"
echo "Start:  $(date)"

python reward_hacking_evaluate/phase2/evaluate_phase2.py \
    --csv "$CSV" \
    --output "$OUT" \
    --concurrency 5

echo "Done:   $(date)"
