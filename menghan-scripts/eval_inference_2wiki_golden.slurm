#!/bin/bash
#SBATCH --job-name=inference-2wiki-golden
#SBATCH --partition=aimi,jjs533,default_partition
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --exclude=dean-compute-02
#SBATCH --time=3:00:00
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/inference_2wiki_golden_%j.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/inference_2wiki_golden_%j.err

eval "$(conda shell.bash hook)"
conda activate lmlm_mx

cd /share/j_sun/mx253/Multi-Hop-Reasoning

MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-4B-SFT_hotpotqa_ep3_bsz48_th-1

echo "========== 2wiki golden contexts + concat-all-db + train params =========="
python menghan-scripts/run_inference_csv.py \
    --model-path "${MODEL_PATH}" \
    --dataset 2wiki \
    --split dev \
    --use-contexts golden \
    --total-count 1000 \
    --batch-size 8 \
    --use-train-params \
    --concat-all-db \
    --similarity-threshold 0.6 \
    --output KG_results/sft_2wiki_dev_n1000_golden_concat_trainparams.csv

echo "Done!"
