#!/bin/bash
#SBATCH --job-name=eval-v2-4b-sft
#SBATCH --partition=aimi,dean,default_partition
#SBATCH --gres=gpu:1
#SBATCH --constraint="a6000|6000ada"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/eval_v2_4b_sft_%j.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/eval_v2_4b_sft_%j.err

eval "$(conda shell.bash hook)"
conda activate lmlm_mx

cd /share/j_sun/mx253/Multi-Hop-Reasoning

MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-4B-SFT_hotpotqa_ep3_bsz48_th-1

mkdir -p /share/j_sun/mx253/Multi-Hop-Reasoning/KG_results_v2/inference

echo "========== 4B SFT — 1000 rows, all-concat, train params, with --log-jsonl =========="
python menghan-scripts/run_inference_csv.py \
    --model-path "${MODEL_PATH}" \
    --dataset hotpotqa \
    --split dev \
    --use-contexts all \
    --total-count 1000 \
    --batch-size 8 \
    --use-train-params \
    --concat-all-db \
    --similarity-threshold 0.6 \
    --output /share/j_sun/mx253/Multi-Hop-Reasoning/KG_results_v2/inference/qwen3_4b_sft_n1000_prod.csv \
    --log-jsonl /share/j_sun/mx253/Multi-Hop-Reasoning/KG_results_v2/inference/qwen3_4b_sft_n1000_prod.jsonl

echo "Done."
