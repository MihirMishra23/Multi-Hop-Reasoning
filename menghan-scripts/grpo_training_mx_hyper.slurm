#!/bin/bash
#SBATCH --job-name=grpo-lmlm
#SBATCH --partition=aimi,jjs533,dean
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/grpo_train_%j.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/grpo_train_%j.err

eval "$(conda shell.bash hook)"
conda activate lmlm_b200

cd /share/j_sun/mx253/Multi-Hop-Reasoning

bash menghan-scripts/grpo_training_mx.sh --gpu_type B200 --two_phase --train_size 1000 --learning_rate 1e-5
