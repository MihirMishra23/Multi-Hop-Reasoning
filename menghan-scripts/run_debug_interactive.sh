#!/bin/bash
# Interactive debug runner for 8B FSDP->vLLM sync issue.
# Run inside salloc'd interactive shell with 2 B200s.
# Triggers _sync_fsdp1_params_to_vllm() and prints [FSDP_TO_VLLM_SYNC] lines.
# Once you see the SUMMARY block, Ctrl+C to cancel.

set -e

cd /share/j_sun/mx253/Multi-Hop-Reasoning

# Make sure conda env is active
if [[ "$CONDA_DEFAULT_ENV" != "lmlm_b200" ]]; then
    echo "Activating lmlm_b200..."
    eval "$(conda shell.bash hook)"
    conda activate lmlm_b200
fi

# Env vars (same as scripts/grpo_train.sh)
export VLLM_USE_V1=0
export VLLM_USE_FLASHINFER=0
export VLLM_TORCH_COMPILE_LEVEL=0
export VLLM_BATCH_INVARIANT=0
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}

# Disable wandb for fast iteration
export WANDB_MODE=disabled

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
echo "Detected ${NUM_GPUS} GPU(s)"

if [[ "${NUM_GPUS}" -lt 2 ]]; then
    echo "ERROR: need at least 2 GPUs for FSDP sharding. Got ${NUM_GPUS}."
    exit 1
fi

MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-8B-SFT_hotpotqa_ep3_bsz48_lr5e-5_th-1

echo ""
echo "==============================================================="
echo "Starting 8B GRPO debug run (will print [FSDP_TO_VLLM_SYNC] lines)"
echo "Once you see the SUMMARY block in output, Ctrl+C to cancel."
echo "==============================================================="
echo ""

# Just call scripts/grpo_train.sh -- it auto-detects 8B and uses fsdp_${NUM_GPUS}.yaml
# Small train_size so it gets through setup quickly to the sync step.
bash scripts/grpo_train.sh \
    --gpu_type B200 \
    --model_path "${MODEL_PATH}" \
    --dataset_name hotpotqa \
    --database_path "" \
    --train_size 100 \
    --total_batch_size 64 \
    --reward_func f1 \
    --phase1_prompt_type sft \
    --two_phase \
  2>&1 | tee /tmp/debug_run_$$.log

echo ""
echo "Log saved to /tmp/debug_run_$$.log"
echo "Grep for sync output:  grep FSDP_TO_VLLM_SYNC /tmp/debug_run_$$.log"
