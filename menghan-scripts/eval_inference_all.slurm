#!/bin/bash
#SBATCH --job-name=inference-all
#SBATCH --partition=aimi,jjs533,default_partition
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --exclude=dean-compute-02
#SBATCH --time=3:00:00
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/inference_all_%j.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/inference_all_%j.err

eval "$(conda shell.bash hook)"
conda activate lmlm_mx

cd /share/j_sun/mx253/Multi-Hop-Reasoning

# MODEL_PATH=/share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak/checkpoint-500
MODEL_PATH=/share/j_sun/lmlm_multihop/checkpoints/main/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.6_top_k_4
echo "========== all contexts + concat-all-db + train params =========="
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
    --output KG_results/new_sft_1.7b_hotpotqa_dev_n1000_all_concat_trainparams.csv

echo "Done!"
