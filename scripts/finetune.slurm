#!/bin/bash
#SBATCH --mail-user=ryan.noonan@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=sft_1.7b
#SBATCH --nodes=1
#SBATCH -p jjs533
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --constraint='6000ada'
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err


set -euo pipefail

# Run from the directory where you submitted the job
cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs
source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate multi-hop-reasoning

export NUM_TRAIN_EPOCHS=5 #change to 5
export NUM_GPUs=1
export PER_DEVICE_TRAIN_BATCH_SIZE=8
export GRADIENT_ACCUMULATION_STEPS=4 #change to 4
export EVAL_ACCUMULATION_STEPS=1 #number of steps before copying metrics to CPU, avoids OOM
export EFFECTIVE_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUs))
export OUTPUT_ROOT="/share/j_sun/rtn27/checkpoints/qwen3-1.7B_sft_v1.3_5743"
export OUTPUT_DIR="${OUTPUT_ROOT}/sft_ep${NUM_TRAIN_EPOCHS}_bsz${EFFECTIVE_BATCH_SIZE}_new_qa_$(date +'%y-%m-%d_%H-%M')"


pwd
echo "Running finetune..."
bash scripts/finetune_w_args.sh \
    /share/j_sun/lmlm_multihop/sft_data/12-19_rollouts_combined_12k_5743_examples_6000_triplets_filtered.json \
    qwen3-1.7b-gemini_sft_v1.3_5743

echo "Finetune complete!"
