#!/bin/bash
# 8B SFT eval — TRAIN PARAMS (T=1.0). Matches GRPO in-loop eval sampling.
# Critical test: if EM here is ~0.35, sync is broken (T=1.0 alone can't explain
# the GRPO 0.15). If EM here is ~0.15, sampling alone explains the drop.
#SBATCH --job-name=eval-8b-sft-trainparams
#SBATCH --partition=aimi,dean,default_partition
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --exclude=jjs533-compute-01,jjs533-compute-02,jjs533-compute-03
#SBATCH --time=3:00:00
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/eval_8b_sft_trainparams_%j.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/eval_8b_sft_trainparams_%j.err

eval "$(conda shell.bash hook)"
conda activate lmlm_b200

cd /share/j_sun/mx253/Multi-Hop-Reasoning

MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-8B-SFT_hotpotqa_ep3_bsz48_lr5e-5_th-1

echo "========== 8B SFT + train params (T=1.0, top_p=0.95, top_k=4) =========="
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
    --save-version "_trainparams" \
    --use-train-params \
    --eval \
    --resume

echo "Done (train params)!"
