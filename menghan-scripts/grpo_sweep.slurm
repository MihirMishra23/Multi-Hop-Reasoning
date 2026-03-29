#!/bin/bash
#SBATCH --job-name=grpo-sweep
#SBATCH --partition=aimi,jjs533,dean
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --exclude=aimi-compute-03
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/grpo_sweep_%A_%a.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/grpo_sweep_%A_%a.err
#SBATCH --array=1

eval "$(conda shell.bash hook)"
conda activate lmlm_b200

cd /share/j_sun/mx253/Multi-Hop-Reasoning

case $SLURM_ARRAY_TASK_ID in
    1) GAS=8;  N=8; K=4; LR=3e-6 ;;
esac

echo "Experiment ${SLURM_ARRAY_TASK_ID}: GAS=${GAS}, N=${N}, K=${K}, LR=${LR}"

bash menghan-scripts/grpo_training_mx.sh \
    --two_phase \
    --train_size 1000 \
    --gradient_accumulation_steps ${GAS} \
    --num_generations ${N} \
    --num_db_rollouts ${K} \
    --learning_rate ${LR}
