#!/bin/bash
#SBATCH --job-name=grpo-lmlm
#SBATCH --partition=jjs533
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/grpo_train_%j.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/grpo_train_%j.err

eval "$(conda shell.bash hook)"
conda activate lmlm_mx

cd /share/j_sun/mx253/Multi-Hop-Reasoning
bash scripts/grpo_train.sh