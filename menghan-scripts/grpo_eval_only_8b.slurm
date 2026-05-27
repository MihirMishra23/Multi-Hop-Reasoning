#!/bin/bash
# Eval-only test on 8B SFT checkpoint via LMLMGRPOTrainer.evaluate().
# Matches the EXACT 8B GRPO training config (4xB200, FSDP, vLLM colocate,
# T=1.0, f1 reward, two_phase, tools) but does NO training.
# If eval reward is ~0.15 -> bug is in trainer's eval path (sync/wrap/etc).
# If eval reward is ~0.5  -> evaluate() path is fine, bug is in training state.

#SBATCH --job-name=grpo-eval-only-8b
#SBATCH --partition=aimi,default_partition
#SBATCH --gres=gpu:nvidia_b200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/grpo_eval_only_8b_%j.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/grpo_eval_only_8b_%j.err

eval "$(conda shell.bash hook)"
conda activate lmlm_b200

cd /share/j_sun/mx253/Multi-Hop-Reasoning

# Same env vars as scripts/grpo_train.sh
export VLLM_USE_V1=0
export VLLM_USE_FLASHINFER=0
export VLLM_TORCH_COMPILE_LEVEL=0
export VLLM_BATCH_INVARIANT=0
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}

MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-8B-SFT_hotpotqa_ep3_bsz48_lr5e-5_th-1
SAVE_DIR=/share/j_sun/lmlm_multihop/checkpoints/debug
OUTPUT_DIR="${SAVE_DIR}/${MODEL_PATH##*/}-eval-only"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
echo "  Detected ${NUM_GPUS} GPU(s)"
ACCEL_CONFIG="configs/accelerate/fsdp_${NUM_GPUS}.yaml"

# Match 8B GRPO training defaults from scripts/grpo_train.sh
# N=32, K=4, M=7, B=8 -> total_per_gen = B*N = 256, per_device=4, grad_accum=16 across 4 GPUs
accelerate launch \
    --num_processes=${NUM_GPUS} \
    --config_file=${ACCEL_CONFIG} \
    src/grpo_eval_only.py \
    --model_path="${MODEL_PATH}" \
    --database_path="" \
    --dataset_name=hotpotqa \
    --output_dir="${OUTPUT_DIR}" \
    --train_size=100 \
    --eval_size=100 \
    --num_generations=32 \
    --num_generations_eval=32 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=32 \
    --gradient_accumulation_steps=16 \
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
    --num_db_rollouts=4 \
    --phase1_db_weight_mode=count \
    --beta=0.0 \
    --learning_rate=5e-6 \
    --loss_type=grpo \
    --temperature=1.0 \
    --top_p=0.95 \
    --top_k=4 \
    --retrieval-threshold 0.6 \
    --do_eval \
    --eval-strategy='no' \
    --logging_steps=1 \
    --report_to=none

echo "Eval-only run complete."
