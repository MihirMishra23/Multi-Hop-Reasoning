#!/bin/bash


# SLURM Directives
#SBATCH -J sft_train                           # Job name
#SBATCH -o /home/lz586/icl/slurm_output/memgpt/sft_train_%j.out   # Standard output file
#SBATCH -e /home/lz586/icl/slurm_output/memgpt/sft_train_%j.err   # Standard error file
#SBATCH -N 1                                    # Number of nodes
#SBATCH -n 1                                    # Number of tasks (or cores)
#SBATCH --get-user-env                          # Retrieve the user's login environment
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G                              # Memory requested (2GB per node)
#SBATCH -t infinite                              # Time limit (2 hours)
#SBATCH --partition=aimi,kilian,jjs533,default_partition          # Request partition
#SBATCH --gres=gpu:nvidia_h100_nvl:1                # Request 1 NVIDIA A6000 GPU


# Load the required environmentt
source activate mem

# Navigate to the working directory
cd /home/lz586/icl/Multi-Hop-Reasoning/


echo "Starting SFT training..."

bash /home/lz586/icl/Multi-Hop-Reasoning/scripts/sft_train.sh --model_size 1.7B --threshold 0.8

echo "SFT complete!"
