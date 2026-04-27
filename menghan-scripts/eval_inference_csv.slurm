#!/bin/bash
#SBATCH --job-name=inference-csv
#SBATCH --partition=aimi,jjs533,dean
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1-10:00:00
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/inference_csv_%j.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/inference_csv_%j.err

eval "$(conda shell.bash hook)"
conda activate lmlm_mx

cd /share/j_sun/mx253/Multi-Hop-Reasoning

MODEL_PATH=Qwen3-4B-SFT_hotpotqa_ep3_bsz48_th-1-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak-ckpt500

# ── Run 1: golden contexts + concat-all-db + train params ────────────────────
echo "========== Run 1: golden contexts =========="
python menghan-scripts/run_inference_csv.py \
    --model-path "${MODEL_PATH}" \
    --dataset hotpotqa \
    --split dev \
    --use-contexts golden \
    --total-count 1000 \
    --batch-size 8 \
    --use-train-params \
    --concat-all-db \
    --similarity-threshold 0.6 \
    --output KG_results/ckpt500_hotpotqa_dev_n1000_golden_concat_trainparams.csv

# ── Run 2: all contexts + concat-all-db + train params ───────────────────────
echo "========== Run 2: all contexts =========="
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
    --output KG_results/ckpt500_hotpotqa_dev_n1000_all_concat_trainparams.csv

echo "Done!"
