#!/bin/bash
#SBATCH --job-name=grpo-sweep2
#SBATCH --partition=aimi,jjs533,dean
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --exclude=aimi-compute-03
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/grpo_sweep2_%A_%a.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/grpo_sweep2_%A_%a.err
#SBATCH --array=0-3

export WANDB_DIR=/share/j_sun/mx253/wandb_tmp
export TMPDIR=/share/j_sun/mx253/tmp

set -x
eval "$(conda shell.bash hook)"
conda activate lmlm_b200

cd /share/j_sun/mx253/Multi-Hop-Reasoning

case $SLURM_ARRAY_TASK_ID in
    0) GAS=64; N=8; K=2; LR=5e-6 ;;
    1) GAS=32; N=8; K=2; LR=5e-6 ;;
    2) GAS=64; N=32; K=4; LR=5e-6 ;;
    3) GAS=32; N=32; K=4; LR=5e-6 ;;
esac

echo "Experiment ${SLURM_ARRAY_TASK_ID}: GAS=${GAS}, N=${N}, K=${K}, LR=${LR}"

bash menghan-scripts/grpo_training_mx.sh \
    --two_phase \
    --train_size 7000 \
    --gradient_accumulation_steps ${GAS} \
    --num_generations ${N} \
    --num_db_rollouts ${K} \
    --learning_rate ${LR}
