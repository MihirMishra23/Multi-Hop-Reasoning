#!/bin/bash
#SBATCH --job-name=sft-lmlm
#SBATCH --partition=aimi
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/sft_train_%j.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/sft_train_%j.err

eval "$(conda shell.bash hook)"
conda activate lmlm_mx

cd /share/j_sun/mx253/Multi-Hop-Reasoning
bash scripts/sft_train.sh