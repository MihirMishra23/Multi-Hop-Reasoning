#!/bin/bash
#SBATCH --job-name=eval-v2-1.7b-grpo
#SBATCH --partition=aimi,dean,default_partition
#SBATCH --gres=gpu:1
#SBATCH --constraint="a6000|6000ada"
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/eval_v2_1.7b_grpo_%j.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/eval_v2_1.7b_grpo_%j.err

eval "$(conda shell.bash hook)"
conda activate lmlm_mx

cd /share/j_sun/mx253/Multi-Hop-Reasoning

MODEL_PATH=/share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak/checkpoint-500

mkdir -p /share/j_sun/mx253/Multi-Hop-Reasoning/KG_results_v2/inference

echo "========== 1.7B GRPO ckpt-500 — 1000 rows, all-concat, train params, with --log-jsonl =========="
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
    --output /share/j_sun/mx253/Multi-Hop-Reasoning/KG_results_v2/inference/qwen3_1.7b_grpo_ckpt500_n1000_prod.csv \
    --log-jsonl /share/j_sun/mx253/Multi-Hop-Reasoning/KG_results_v2/inference/qwen3_1.7b_grpo_ckpt500_n1000_prod.jsonl

echo "Done."
