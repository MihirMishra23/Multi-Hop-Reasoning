#!/bin/bash
# 1-GPU 8B GRPO smoke test as sbatch job.
# See grpo_debug_1gpu_8b.sh for the rationale (single B200, no FSDP).
#
# Submit:  sbatch menghan-scripts/grpo_debug_1gpu_8b.slurm

#SBATCH --job-name=grpo-debug-1gpu-8b
#SBATCH --partition=aimi,default_partition
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --exclude=jjs533-compute-01,jjs533-compute-02,jjs533-compute-03
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/grpo_debug_1gpu_8b_%j.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/grpo_debug_1gpu_8b_%j.err

set -e

eval "$(conda shell.bash hook)"
conda activate lmlm_b200

cd /share/j_sun/mx253/Multi-Hop-Reasoning

# Env (same as scripts/grpo_train.sh)
export VLLM_USE_V1=0
export VLLM_USE_FLASHINFER=0
export VLLM_TORCH_COMPILE_LEVEL=0
export VLLM_BATCH_INVARIANT=0
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}
export WANDB_ENTITY=ryan-noonan-cornell-university
export WANDB_PROJECT=LMLM-Multihop
# do NOT set WANDB_NAME — let grpo_train.py auto-generate from OUTPUT_DIR basename
# (same naming convention as the original 8B GRPO run, just with -debug-1gpu suffix)

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
echo "Detected ${NUM_GPUS} GPU(s)"

MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-8B-SFT_hotpotqa_ep3_bsz48_lr5e-5_th-1
SAVE_DIR=/share/j_sun/lmlm_multihop/checkpoints/debug

# Plain single-GPU (no FSDP)
ACCEL_CONFIG="configs/accelerate/multi_gpu_1.yaml"

# Tiny per-device, smaller TBS, fewer rollouts
PER_DEVICE_TRAIN_BATCH_SIZE=1
NUM_GENERATIONS=8
NUM_DB_ROLLOUTS=4
TOTAL_BATCH_SIZE=64
LEARNING_RATE=5e-6
BETA=0.0
MAX_STEPS=20
TRAIN_SIZE=1000
REWARD_FUNC=f1
PHASE1_PROMPT_TYPE=sft
PHASE1_DB_WEIGHT_MODE=count
RETRIEVAL_THRESHOLD=0.6
TOP_K=4

# Derived (match grpo_train.sh naming)
GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BATCH_SIZE / PER_DEVICE_TRAIN_BATCH_SIZE / NUM_GPUS))
B=$((TOTAL_BATCH_SIZE / NUM_GENERATIONS))
M=$(((NUM_GENERATIONS - NUM_DB_ROLLOUTS) / NUM_DB_ROLLOUTS))

# Build OUTPUT_DIR with same format as scripts/grpo_train.sh, append -debug-1gpu
OUTPUT_DIR="${SAVE_DIR}/${MODEL_PATH##*/}-grpo-tbs${TOTAL_BATCH_SIZE}-N${NUM_GENERATIONS}-K${NUM_DB_ROLLOUTS}-B${B}-M${M}-b${BETA}-lr${LEARNING_RATE}-step${MAX_STEPS}-n${TRAIN_SIZE}-${REWARD_FUNC}-2ph-pr${PHASE1_PROMPT_TYPE}-w${PHASE1_DB_WEIGHT_MODE}-th${RETRIEVAL_THRESHOLD}-topk${TOP_K}-nak-debug-1gpu"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "  TBS=${TOTAL_BATCH_SIZE}  PER_DEVICE=${PER_DEVICE_TRAIN_BATCH_SIZE}  GRAD_ACCUM=${GRADIENT_ACCUMULATION_STEPS}"

accelerate launch \
    --num_processes=${NUM_GPUS} \
    --config_file=${ACCEL_CONFIG} \
    src/grpo_train.py \
    --model_path="${MODEL_PATH}" \
    --database_path="" \
    --dataset_name=hotpotqa \
    --output_dir="${OUTPUT_DIR}" \
    --train_size=${TRAIN_SIZE} \
    --eval_size=50 \
    --max_steps=${MAX_STEPS} \
    --num_generations=${NUM_GENERATIONS} \
    --num_generations_eval=${NUM_GENERATIONS} \
    --num_db_rollouts=${NUM_DB_ROLLOUTS} \
    --per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --per_device_eval_batch_size=${NUM_GENERATIONS} \
    --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    --max_completion_length=1024 \
    --vllm_max_model_length=4096 \
    --vllm_gpu_memory_utilization=0.15 \
    --use_vllm \
    --vllm_mode=colocate \
    --tools \
    --gradient_checkpointing \
    --two_phase \
    --reward_func=${REWARD_FUNC} \
    --phase1_reward_type=binary \
    --phase1_prompt_type=${PHASE1_PROMPT_TYPE} \
    --phase1_db_weight_mode=${PHASE1_DB_WEIGHT_MODE} \
    --beta=${BETA} \
    --learning_rate=${LEARNING_RATE} \
    --loss_type=grpo \
    --max_grad_norm=1.0 \
    --warmup_ratio=0.1 \
    --lr_scheduler_type=cosine \
    --temperature=1.0 \
    --top_p=0.95 \
    --top_k=${TOP_K} \
    --retrieval-threshold ${RETRIEVAL_THRESHOLD} \
    --do_eval \
    --eval_strategy='no' \
    --save_strategy='no' \
    --logging_steps=1 \
    --report_to=wandb

echo "Run complete."