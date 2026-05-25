#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEARCH_R1_DIR="${SCRIPT_DIR}/Search-R1"
DATA_DIR="${SCRIPT_DIR}/data"

# Configure HuggingFace cache directory (optional)
# Uncomment and modify if you want to use a custom cache location:
# export HF_HOME=/path/to/cache/huggingface
# export TRANSFORMERS_CACHE=${HF_HOME}
# export HF_DATASETS_CACHE=${HF_HOME}/datasets

# ── Default Configuration ───────────────────────────────────────────
MODEL_SIZE="1.7B"
MODEL_PATH=""
OUTPUT_DIR=""
NUM_GPUS=""
RETRIEVER_URL="http://localhost:8000/retrieve"

# ── Environment Variables ───────────────────────────────────────────
export VLLM_USE_V1=0
export VLLM_USE_FLASHINFER=0
export VLLM_TORCH_COMPILE_LEVEL=0
export VLLM_BATCH_INVARIANT=0
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}
export PYTORCH_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=XFORMERS

# Configure Weights & Biases logging (optional)
# Set your wandb entity and project:
# export WANDB_ENTITY=your-wandb-username
export WANDB_PROJECT=search-r1-hotpotqa

# ── Argument Parsing ────────────────────────────────────────────────
function usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --model_path PATH       Path to base model (default: Qwen/Qwen3-{size})"
    echo "  --model_size SIZE       Model size: 1.7B or 4B (default: 1.7B)"
    echo "  --output_dir DIR        Output directory (default: auto-generated)"
    echo "  --num_gpus N            Number of GPUs (default: auto-detect)"
    echo "  --retriever_url URL     Retriever server URL (default: http://localhost:8000/retrieve)"
    echo ""
    echo "Examples:"
    echo "  $0 --model_size 1.7B"
    echo "  $0 --model_size 4B --num_gpus 8"
    echo "  $0 --model_path /path/to/model --model_size 1.7B"
    echo ""
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --model_size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --retriever_url)
            RETRIEVER_URL="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# Set default model path based on model size if not provided
if [ -z "${MODEL_PATH}" ]; then
    if [ "${MODEL_SIZE}" = "1.7B" ]; then
        MODEL_PATH="Qwen/Qwen3-1.7B"
    elif [ "${MODEL_SIZE}" = "4B" ]; then
        MODEL_PATH="Qwen/Qwen3-4B"
    else
        echo "❌ Error: Unsupported model size: ${MODEL_SIZE}"
        echo "Supported sizes: 1.7B, 4B"
        exit 1
    fi
    echo "ℹ️  Using default model path: ${MODEL_PATH}"
fi

# Auto-detect number of GPUs if not specified
if [ -z "${NUM_GPUS}" ]; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    if [ "${NUM_GPUS}" -eq 0 ]; then
        NUM_GPUS=1
    fi
fi

# Generate output directory if not specified
if [ -z "${OUTPUT_DIR}" ]; then
    MODEL_NAME=$(basename "${MODEL_PATH}" | sed 's/\//-/g')
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${SCRIPT_DIR}/output/${MODEL_NAME}_searchr1_hotpotqa_${TIMESTAMP}"
fi

# Set CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
export CUDA_VISIBLE_DEVICES

# Create experiment name
EXPERIMENT_NAME="hotpotqa-search-r1-${MODEL_SIZE}-f1"

echo "════════════════════════════════════════════════════════════════"
echo "Search-R1 Training on HotpotQA"
echo "════════════════════════════════════════════════════════════════"
echo "Model:       ${MODEL_PATH}"
echo "Size:        ${MODEL_SIZE}"
echo "Output:      ${OUTPUT_DIR}"
echo "GPUs:        ${NUM_GPUS} (${CUDA_VISIBLE_DEVICES})"
echo "Retriever:   ${RETRIEVER_URL}"
echo "Experiment:  ${EXPERIMENT_NAME}"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check if data exists
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "❌ Error: Training data not found at ${DATA_DIR}/train.parquet"
    echo "Please run prepare_hotpotqa_data.py first"
    exit 1
fi

# Check if Search-R1 is installed
if [ ! -d "${SEARCH_R1_DIR}" ]; then
    echo "❌ Error: Search-R1 not found at ${SEARCH_R1_DIR}"
    echo "Please run setup_environment.sh first"
    exit 1
fi

# Activate conda environment (override with CONDA_ENV=...)
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-mem-searchr1}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo "Starting Search-R1 training with verl..."
echo ""

cd "${SEARCH_R1_DIR}"

# Launch training using Search-R1's main_ppo with Hydra config
# Based on train_ppo.sh from Search-R1 repo, adapted for HotpotQA
RAY_BACKEND_LOG_LEVEL=debug PYTHONUNBUFFERED=1 python3 -u -m verl.trainer.main_ppo \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=512 \
    data.val_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.grad_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=32 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=False \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path="${MODEL_PATH}" \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=8 \
    critic.model.fsdp_config.param_offload=false \
    critic.model.fsdp_config.grad_offload=false \
    critic.model.fsdp_config.optimizer_offload=false \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_epochs=15 \
    trainer.total_training_steps=500 \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    max_turns=4 \
    retriever.url="${RETRIEVER_URL}" \
    retriever.topk=3 \
    2>&1 | tee "${OUTPUT_DIR}/${EXPERIMENT_NAME}.log"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✓ Training complete!"
echo "════════════════════════════════════════════════════════════════"
echo "Output saved to: ${OUTPUT_DIR}"
echo "Log file: ${OUTPUT_DIR}/${EXPERIMENT_NAME}.log"
echo ""
