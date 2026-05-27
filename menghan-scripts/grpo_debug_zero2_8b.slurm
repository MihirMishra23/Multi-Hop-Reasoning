#!/bin/bash
# 4-GPU 8B GRPO with DeepSpeed Zero-2 (instead of FSDP).
# Isolation test: if reward jumps to ~0.5, FSDP path is the culprit.
# If reward still ~0.18, bug is NOT FSDP-specific.
#
# Submit:  sbatch menghan-scripts/grpo_debug_zero2_8b.slurm

#SBATCH --job-name=grpo-debug-zero2-8b
#SBATCH --partition=aimi,default_partition
#SBATCH --gres=gpu:nvidia_b200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --exclude=jjs533-compute-01,jjs533-compute-02,jjs533-compute-03
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/grpo_debug_zero2_8b_%j.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/grpo_debug_zero2_8b_%j.err

set -e

eval "$(conda shell.bash hook)"
conda activate lmlm_b200

cd /share/j_sun/mx253/Multi-Hop-Reasoning

# Sanity check deepspeed is installed
python -c "import deepspeed" 2>/dev/null || {
    echo "ERROR: deepspeed not installed in lmlm_b200 env. Run: pip install deepspeed"
    exit 1
}
python -c "import deepspeed; print(f'deepspeed version: {deepspeed.__version__}')"

# Env (same as scripts/grpo_train.sh)
export VLLM_USE_V1=0
export VLLM_USE_FLASHINFER=0
export VLLM_TORCH_COMPILE_LEVEL=0
export VLLM_BATCH_INVARIANT=0
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}
export WANDB_NAME="debug-zero2-8b-${SLURM_JOB_ID}"
echo "WANDB_NAME=${WANDB_NAME}"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
echo "Detected ${NUM_GPUS} GPU(s)"

MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-8B-SFT_hotpotqa_ep3_bsz48_lr5e-5_th-1
SAVE_DIR=/share/j_sun/lmlm_multihop/checkpoints/debug
OUTPUT_DIR="${SAVE_DIR}/${MODEL_PATH##*/}-zero2-debug"

# Use Zero-2 instead of FSDP
ACCEL_CONFIG="configs/accelerate/zero2_${NUM_GPUS}.yaml"
if [[ ! -f "${ACCEL_CONFIG}" ]]; then
    echo "ERROR: ${ACCEL_CONFIG} not found"
    exit 1
fi

# Same as 8B FSDP config (matches grpo_train.sh 8B preset)
PER_DEVICE_TRAIN_BATCH_SIZE=4
NUM_GENERATIONS=32
NUM_DB_ROLLOUTS=4
TOTAL_BATCH_SIZE=512
GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BATCH_SIZE / PER_DEVICE_TRAIN_BATCH_SIZE / NUM_GPUS))
echo "  TBS=${TOTAL_BATCH_SIZE}  PER_DEVICE=${PER_DEVICE_TRAIN_BATCH_SIZE}  GRAD_ACCUM=${GRADIENT_ACCUMULATION_STEPS}"

accelerate launch \
    --num_processes=${NUM_GPUS} \
    --config_file=${ACCEL_CONFIG} \
    src/grpo_train.py \
    --model_path="${MODEL_PATH}" \
    --database_path="" \
    --dataset_name=hotpotqa \
    --output_dir="${OUTPUT_DIR}" \
    --train_size=1000 \
    --eval_size=50 \
    --max_steps=20 \
    --num_generations=${NUM_GENERATIONS} \
    --num_generations_eval=${NUM_GENERATIONS} \
    --num_db_rollouts=${NUM_DB_ROLLOUTS} \
    --per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size=${NUM_GENERATIONS} \
    --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    --max_completion_length=1024 \
    --vllm_max_model_length=4096 \
    --vllm_gpu_memory_utilization=0.2 \
    --use_vllm \
    --vllm_mode=colocate \
    --tools \
    --gradient_checkpointing \
    --two_phase \
    --reward_func=f1 \
    --phase1_reward_type=binary \
    --phase1_prompt_type=sft \
    --phase1_db_weight_mode=count \
    --beta=0.0 \
    --learning_rate=5e-6 \
    --loss_type=grpo \
    --max_grad_norm=1.0 \
    --warmup_ratio=0.1 \
    --lr_scheduler_type=cosine \
    --temperature=1.0 \
    --top_p=0.95 \
    --top_k=4 \
    --retrieval-threshold 0.6 \
    --do_eval \
    --eval_strategy='no' \
    --save_strategy='no' \
    --logging_steps=1 \
    --report_to=wandb

echo "Run complete."