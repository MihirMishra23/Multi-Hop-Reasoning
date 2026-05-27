#!/bin/bash
#SBATCH --job-name=sanity-logjsonl
#SBATCH --partition=aimi,dean,default_partition
#SBATCH --gres=gpu:1
#SBATCH --constraint="a6000|6000ada"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0:30:00
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/sanity_logjsonl_%j.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/sanity_logjsonl_%j.err

eval "$(conda shell.bash hook)"
conda activate lmlm_mx

cd /share/j_sun/mx253/Multi-Hop-Reasoning

# 1.7B SFT (fastest model) for sanity check
MODEL_PATH=/share/j_sun/lmlm_multihop/checkpoints/main/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.6_top_k_4

mkdir -p /share/j_sun/mx253/Multi-Hop-Reasoning/KG_results_v2

echo "========== SANITY: n=10, with --log-jsonl =========="
python menghan-scripts/run_inference_csv.py \
    --model-path "${MODEL_PATH}" \
    --dataset hotpotqa \
    --split dev \
    --use-contexts all \
    --total-count 10 \
    --batch-size 5 \
    --use-train-params \
    --concat-all-db \
    --similarity-threshold 0.6 \
    --output /share/j_sun/mx253/Multi-Hop-Reasoning/KG_results_v2/sanity_1.7b_sft_n10.csv \
    --log-jsonl /share/j_sun/mx253/Multi-Hop-Reasoning/KG_results_v2/sanity_1.7b_sft_n10.jsonl

echo "Done. Now verify:"
echo "  /share/j_sun/mx253/Multi-Hop-Reasoning/KG_results_v2/sanity_1.7b_sft_n10.csv"
echo "  /share/j_sun/mx253/Multi-Hop-Reasoning/KG_results_v2/sanity_1.7b_sft_n10.jsonl"
