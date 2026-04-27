#!/bin/bash

# Launch multiple SFT training jobs with different configurations

echo "Submitting 4 SFT training jobs..."
echo ""

# Job 1: 4B with threshold 0.9
sbatch --job-name=sft-4b-0.9 \
       --export=ALL,MODEL_SIZE=4B,THRESHOLD=0.9 \
       scripts/sbatch/sft_training.slurm
echo "✓ Submitted: sft-4b-0.9 (model_size=4B, threshold=0.9)"

# Job 2: 4B without threshold
sbatch --job-name=sft-4b \
       --export=ALL,MODEL_SIZE=4B \
       scripts/sbatch/sft_training.slurm
echo "✓ Submitted: sft-4b (model_size=4B, no threshold)"

# Job 3: 1.7B with threshold 0.9
sbatch --job-name=sft-1.7b-0.9 \
       --export=ALL,MODEL_SIZE=1.7B,THRESHOLD=0.9 \
       scripts/sbatch/sft_training.slurm
echo "✓ Submitted: sft-1.7b-0.9 (model_size=1.7B, threshold=0.9)"

# Job 4: 1.7B without threshold
sbatch --job-name=sft-1.7b \
       --export=ALL,MODEL_SIZE=1.7B \
       scripts/sbatch/sft_training.slurm
echo "✓ Submitted: sft-1.7b (model_size=1.7B, no threshold)"

echo ""
echo "All jobs submitted! Check status with: squeue -u $USER"
