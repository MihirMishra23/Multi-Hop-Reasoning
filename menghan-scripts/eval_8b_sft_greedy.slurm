#!/bin/bash
# 8B SFT eval — GREEDY (T=0). Establishes the model's best-case ceiling.
#SBATCH --job-name=eval-8b-sft-greedy
#SBATCH --partition=aimi,dean,default_partition
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --exclude=jjs533-compute-01,jjs533-compute-02,jjs533-compute-03
#SBATCH --time=3:00:00
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/eval_8b_sft_greedy_%j.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/eval_8b_sft_greedy_%j.err

eval "$(conda shell.bash hook)"
conda activate lmlm_b200

cd /share/j_sun/mx253/Multi-Hop-Reasoning

MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-8B-SFT_hotpotqa_ep3_bsz48_lr5e-5_th-1

echo "========== 8B SFT + greedy (T=0) =========="
python src/eval_multihop.py \
    --model-path "${MODEL_PATH}" \
    --method two_phase \
    --dataset hotpotqa \
    --split dev \
    --setting distractor \
    --use-contexts all \
    --concat-all-db \
    --total-count 300 \
    --batch-size 64 \
    --seed 42 \
    --save-every 64 \
    --start-index 0 \
    --top-k 4 \
    --similarity-threshold 0.6 \
    --use-inverses \
    --output-dir ./output/debug_8b_eval/ \
    --save-version "_greedy" \
    --eval \
    --resume

echo "Done (greedy)!"
