#!/bin/bash


# SLURM Directives
#SBATCH -J sft_train_qwen2.5_7b                 # Job name
#SBATCH -o /home/lz586/icl/slurm_output/lmlm_multihop/sft_train_qwen2.5_7b_%j.out
#SBATCH -e /home/lz586/icl/slurm_output/lmlm_multihop/sft_train_qwen2.5_7b_%j.err
#SBATCH -N 1                                    # Number of nodes
#SBATCH -n 1                                    # Number of tasks (or cores)
#SBATCH --get-user-env                          # Retrieve the user's login environment
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G                              # Memory requested
#SBATCH -t infinite                             # Time limit
#SBATCH --partition=aimi,kilian,jjs533,default_partition
#SBATCH --gres=gpu:nvidia_b200:1                # 1 NVIDIA B200 GPU (180GB fits 7B)


# Load the required environment
source activate mem

# Navigate to the working directory
cd /home/lz586/icl/Multi-Hop-Reasoning/


echo "Starting SFT training (Qwen2.5-7B ablation)..."

# Matches Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1 hyperparameters
# (eff bsz 48 = 6 * 8, ep=3, lr=5e-5, threshold=-1, two-phase).
# seq=1024 mirrors the 4B/8B SFT defaults to keep activation memory in check.
bash /home/lz586/icl/Multi-Hop-Reasoning/scripts/sft_train.sh \
    --model_size 7B \
    --threshold -1 \
    --dataset hotpotqa \
    --two_phase True \
    --max_seq_length 1024

echo "SFT complete!"
