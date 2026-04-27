#!/bin/bash


# SLURM Directives
#SBATCH -J generate_tier                           # Job name
#SBATCH -o /home/lz586/icl/slurm_output/lmlm_multihop/generate_tier_%j.out
#SBATCH -e /home/lz586/icl/slurm_output/lmlm_multihop/generate_tier_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=28
#SBATCH --mem=256G
#SBATCH -t 3-00:00:00
#SBATCH --partition=aimi
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --requeue


# Load the required environmentt
source activate mem

# Navigate to the working directory
cd /home/lz586/icl/Multi-Hop-Reasoning/

echo "Generating tier..."
bash /home/lz586/icl/Multi-Hop-Reasoning/scripts/generate_tier.sh \
    --dataset hotpotqa \
    --split train_full \
    --num_samples 90447 \
    --num_rollouts 4 \
    --answer_threshold 0.6 \
    --two_phase \
    --chunk_size 6000 \
    --model_path /share/j_sun/lmlm_multihop/checkpoints/main/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.6_top_k_4

echo "Tier generation complete!"
