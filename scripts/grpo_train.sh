#!/bin/bash


# SLURM Directives
#SBATCH -J grpo_train                           # Job name
#SBATCH -o /home/as2637/lmlm_debug/Multi-Hop-Reasoning/grpo_train_%j.out   # Standard output file
#SBATCH -e /home/as2637/lmlm_debug/Multi-Hop-Reasoning/grpo_train_%j.err   # Standard error file
#SBATCH -N 1                                    # Number of nodes
#SBATCH -n 1                                    # Number of tasks (or cores)
#SBATCH --get-user-env                          # Retrieve the user's login environment
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G                              # Memory requested (2GB per node)
#SBATCH -t infinite                              # Time limit (2 hours)
#SBATCH --partition=aimi,jjs533,default_partition          # Request partition
#SBATCH --gres=gpu:nvidia_b200:1                # Request 1 NVIDIA A6000 GPU
#SBATCH --requeue


# Load the required environmentt
source activate lmlm_multihop 

# GRPO Training Script for LMLM Multi-Hop QA
export CUDA_LAUNCH_BLOCKING=1
# 1 breaks training backward (vLLM overrides aten::mm → aten::linear_backward has no CUDA kernel)
export VLLM_BATCH_INVARIANT=0
export TORCH_USE_CUDA_DSA=1
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# export WANDB_ENTITY=ryan-noonan-cornell-university
# export WANDB_PROJECT=LMLM-Multihop

export WANDB_ENTITY=as2637-cornell-university   
export WANDB_PROJECT=LMLM-Multihop

# DEBUG
export VLLM_ATTENTION_BACKEND=TRITON_ATTN   # with batch_invariant=0; FLASH_ATTN→PTX error on some clusters
export VLLM_USE_FLASHINFER=0


# bash /home/lz586/icl/Multi-Hop-Reasoning/scripts/grpo_train.sh --gpu_type H100
# Default values
GPU_TYPE="B200"
# MODEL_PATH="Qwen/Qwen3-1.7B"
MODEL_PATH=/share/j_sun/rtn27/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep5_bsz32_th-1
DATABASE_PATH="/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_train_start_idx_82347_nb_8100_database.json"
# DATABASE_PATH="" # -> Not used for two phase
#SAVE_DIR=/share/j_sun/lmlm_multihop/checkpoints/debug

SAVE_DIR=/share/j_sun/as2637/lmlm_multihop/checkpoints/debug
DATASET_NAME="hotpotqa"
NUM_GPUS=2

LOSS_TYPE="grpo"
VLLM_GPU_MEMORY_UTILIZATION=0.15
BETA=0.0
LEARNING_RATE=1e-6
NUM_GENERATIONS=4
NUM_TRAIN_EPOCHS=5 # default 3
TRAIN_SIZE=700
EVAL_SIZE=100
TRAIN_DATA_PATH=""
EVAL_DATA_PATH=""
MAX_COMPLETION_LENGTH=1024
EVAL_STEPS=1
LOGGING_STEPS=5
TOP_P=0.95
TEMPERATURE=1.3
TOP_K=0
IS_ADAPTIVE_K=False
RETRIEVAL_THRESHOLD=0.9
REWARD_FUNC="em_size"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu_type)
            GPU_TYPE="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --database_path)
            DATABASE_PATH="$2"
            shift 2
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --num_train_epochs)
            NUM_TRAIN_EPOCHS="$2"
            shift 2
            ;;
        --is_adaptive_k)
            IS_ADAPTIVE_K="$2"
            shift 2
            ;;
        --two_phase)
            TWO_PHASE=1
            shift 1
            ;;
        --train_size)
            TRAIN_SIZE="$2"
            shift 2
            ;;
        --debug)
            DEBUG=1
            shift 1
            ;;
        --reward_func)
            REWARD_FUNC="$2"
            shift 2
            ;;
        --train_data_path)
            TRAIN_DATA_PATH="$2"
            shift 2
            ;;
        --eval_data_path)
            EVAL_DATA_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ "$GPU_TYPE" == "B200" ]; then
    # B200
    if [[ "${MODEL_PATH}" == *"1.7B"* ]]; then
        NUM_GPUS=2
        NUM_GENERATIONS=8
        PER_DEVICE_TRAIN_BATCH_SIZE=4
        GRADIENT_ACCUMULATION_STEPS=8
        VLLM_GPU_MEMORY_UTILIZATION=0.15
    elif [[ "${MODEL_PATH}" == *"4B"* ]]; then
        NUM_GPUS=2
        NUM_GENERATIONS=8
        PER_DEVICE_TRAIN_BATCH_SIZE=4
        GRADIENT_ACCUMULATION_STEPS=8
        VLLM_GPU_MEMORY_UTILIZATION=0.15
    elif [[ "${MODEL_PATH}" == *"382M"* ]]; then
        NUM_GPUS=1
        NUM_GENERATIONS=8
        PER_DEVICE_TRAIN_BATCH_SIZE=256
        GRADIENT_ACCUMULATION_STEPS=1
        VLLM_GPU_MEMORY_UTILIZATION=0.15
    else
        echo "Invalid model path: ${MODEL_PATH}"
        exit 1
    fi
elif [ "$GPU_TYPE" == "H100" ]; then
    # H100 debug
    NUM_GPUS=2
    NUM_GENERATIONS=8
    PER_DEVICE_TRAIN_BATCH_SIZE=4
    GRADIENT_ACCUMULATION_STEPS=8
    VLLM_GPU_MEMORY_UTILIZATION=0.2
fi

if [ "$NUM_GPUS" == 2 ]; then
    export CUDA_VISIBLE_DEVICES=0,1
elif [ "$NUM_GPUS" == 1 ]; then
    export CUDA_VISIBLE_DEVICES=0
else
    echo "Invalid number of GPUs: ${NUM_GPUS}"
    exit 1
fi

if [ -n "${DEBUG}" ]; then
    echo "Debug mode enabled"
    TRAIN_SIZE=100
    EVAL_SIZE=10
    NUM_TRAIN_EPOCHS=10
    NUM_GPUS=1

    NUM_GENERATIONS=8
    GRADIENT_ACCUMULATION_STEPS=8
    PER_DEVICE_TRAIN_BATCH_SIZE=8
    VLLM_GPU_MEMORY_UTILIZATION=0.15
fi


#-------------------------------- Save dir --------------------------------
# output_dir = script_args.model_path.split('/')[-1]+'-'+str(grpo_config.loss_type)+'-g'+str(grpo_config.num_generations)+'-bs'+str(grpo_config.per_device_train_batch_size)+'-s'+str(grpo_config.gradient_accumulation_steps)+'-b'+str(grpo_config.beta)+'-ep'+str(grpo_config.num_train_epochs)+'-n'+str(script_args.train_size)
OUTPUT_DIR="${SAVE_DIR}/${MODEL_PATH##*/}-${LOSS_TYPE}-g${NUM_GENERATIONS}-bs${PER_DEVICE_TRAIN_BATCH_SIZE}-s${GRADIENT_ACCUMULATION_STEPS}-b${BETA}-ep${NUM_TRAIN_EPOCHS}-n${TRAIN_SIZE}-${REWARD_FUNC}"
if [ -n "${TWO_PHASE}" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}-v2"
    TWO_PHASE="--two_phase"
else
    TWO_PHASE=""
fi
if [ -n "${DEBUG}" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}-debug"
fi

# Use it in training
LAST_CKPT=$(ls -d $OUTPUT_DIR/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
if [ -n "$LAST_CKPT" ]; then
    RESUME_FROM_CHECKPOINT="--resume_from_checkpoint=${LAST_CKPT}"
else
    RESUME_FROM_CHECKPOINT=""
fi

# "${MODEL_NAME_OR_PATH##*/}-SFT_ep${NUM_TRAIN_EPOCHS}_bsz${EFFECTIVE_BATCH_SIZE}_th${THRESHOLD}"

# split the threshold by split the _th from the model path
BASENAME="${MODEL_PATH##*/}"
THRESHOLD="${BASENAME##*_th}"

if [ "${THRESHOLD}" = "-3" ]; then
    RETURN_TRIPLES="--return_triples"
    echo "RETURN_TRIPLES: ${RETURN_TRIPLES}"
else
    RETURN_TRIPLES=""
fi

if [ "${IS_ADAPTIVE_K}" = "True" ]; then
    ADAPTIVE_K="--adaptive_k"
    echo "ADAPTIVE_K: ${ADAPTIVE_K}"
else
    ADAPTIVE_K=""
    OUTPUT_DIR="${OUTPUT_DIR}-nak"
fi

echo "Starting GRPO training with:"
echo "  Model: ${MODEL_PATH}"
echo "  Database: ${DATABASE_PATH}"
echo "  Output: ${OUTPUT_DIR}"
echo "  GPUs: ${NUM_GPUS}"
echo "  GPU Type: ${GPU_TYPE}"
echo "  Resume from checkpoint: ${RESUME_FROM_CHECKPOINT}"
echo "  Two phase: ${TWO_PHASE}"
echo "  Return triples: ${RETURN_TRIPLES}"
echo "  Adaptive k: ${ADAPTIVE_K}"


accelerate launch \
  --num_processes=${NUM_GPUS} \
  --config_file=configs/accelerate/multi_gpu_${NUM_GPUS}.yaml \
  src/grpo_train.py \
  --model_path="${MODEL_PATH}" \
  --dataset_name="${DATASET_NAME}" \
  --database_path="${DATABASE_PATH}" \
  --output_dir="${OUTPUT_DIR}" \
  --num_generations=${NUM_GENERATIONS} \
  --num_generations_eval=${NUM_GENERATIONS} \
  --per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
  --gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
  --max_completion_length=${MAX_COMPLETION_LENGTH} \
  --eval_steps=${EVAL_STEPS} \
  --logging_steps=${LOGGING_STEPS} \
  --vllm_gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION} \
  --use_vllm \
  --vllm_mode=colocate \
  --tools \
  --gradient_checkpointing \
  --do_eval \
  --log_completions \
  --beta=${BETA} \
  --learning_rate=${LEARNING_RATE} \
  --loss_type=${LOSS_TYPE} \
  --max_grad_norm=1.0 \
  --warmup_ratio=0.1 \
  --vllm_max_model_length=4096 \
  --train_size=${TRAIN_SIZE} \
  --eval_size=${EVAL_SIZE} \
  $([ -n "${TRAIN_DATA_PATH}" ] && echo "--train_data_path=${TRAIN_DATA_PATH}") \
  $([ -n "${EVAL_DATA_PATH}" ] && echo "--eval_data_path=${EVAL_DATA_PATH}") \
  --top_p=${TOP_P} \
  --temperature=${TEMPERATURE} \
  --top_k=${TOP_K} \
  --num_train_epochs=${NUM_TRAIN_EPOCHS} \
  --save_strategy=steps \
  --save_total_limit=5 \
  --save_steps=0.2 \
  ${RESUME_FROM_CHECKPOINT} \
  --use-inverses \
  --retrieval-threshold ${RETRIEVAL_THRESHOLD} \
  ${TWO_PHASE} \
  ${RETURN_TRIPLES} \
  ${ADAPTIVE_K} \
  --reward_func=${REWARD_FUNC}

echo "Training completed!"