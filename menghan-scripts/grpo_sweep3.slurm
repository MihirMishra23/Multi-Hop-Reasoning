#!/bin/bash
#SBATCH --job-name=grpo-n7k
#SBATCH --partition=aimi,jjs533,dean
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --exclude=aimi-compute-03
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/grpo_n7k_%j.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/grpo_n7k_%j.err

set -ex
eval "$(conda shell.bash hook)"
conda activate lmlm_b200

cd /share/j_sun/mx253/Multi-Hop-Reasoning

bash menghan-scripts/grpo_training_mx.sh \
    --two_phase \
    --train_size 7000 \
    --gradient_accumulation_steps 32 \
    --num_generations 16 \
    --num_db_rollouts 4 \
    --learning_rate 5e-6
