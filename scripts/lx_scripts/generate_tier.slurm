#!/bin/bash


# SLURM Directives
#SBATCH -J generate_tier                           # Job name
#SBATCH -o /home/lz586/icl/slurm_output/lmlm_multihop/generate_tier_%j.out   # Standard output file
#SBATCH -e /home/lz586/icl/slurm_output/lmlm_multihop/generate_tier_%j.err   # Standard error file
#SBATCH -N 1                                    # Number of nodes
#SBATCH -n 1                                    # Number of tasks (or cores)
#SBATCH --get-user-env                          # Retrieve the user's login environment
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G                              # Memory requested (2GB per node)
#SBATCH -t infinite                              # Time limit (2 hours)
#SBATCH --partition=aimi,kilian,jjs533,default_partition          # Request partition
#SBATCH --gres=gpu:nvidia_rtx_a6000:1                # Request 1 NVIDIA A6000 GPU
#SBATCH --requeue


# Load the required environmentt
source activate mem

# Navigate to the working directory
cd /home/lz586/icl/Multi-Hop-Reasoning/


bash /home/lz586/icl/Multi-Hop-Reasoning/scripts/generate_tier.sh

echo "Tier generation complete!"
