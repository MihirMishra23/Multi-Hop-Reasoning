#!/bin/bash


# ── Diagnostic slurm ──────────────────────────────────────────────────────────
# Reproduces the running 3B GRPO setup (job 970679) but FORCES FSDP.
# Purpose: isolate whether FSDP+vLLM colocate weight sync is broken.
# Expected outcomes after the first reward log (~3-5 min after rollout start):
#   - reward ≈ 0.5-0.6, ISR ≈ 0.3-0.5  →  FSDP works; 7B problem is elsewhere
#   - reward ≈ 0.1-0.2, ISR ≈ 0.03-0.10  →  FSDP weight sync is broken (matches 7B)
# scancel immediately after the verdict is in.
#
# SLURM Directives
#SBATCH -J grpo_3b_fsdp_probe                    # Job name
#SBATCH -o /home/lz586/icl/slurm_output/lmlm_multihop/grpo_3b_fsdp_probe_%j.out
#SBATCH -e /home/lz586/icl/slurm_output/lmlm_multihop/grpo_3b_fsdp_probe_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --get-user-env
#SBATCH --cpus-per-task=28
#SBATCH --mem=256G
#SBATCH -t 1:00:00                               # 1 h is plenty for the probe
#SBATCH --partition=aimi,kilian,jjs533,default_partition
#SBATCH --gres=gpu:nvidia_b200:4
#SBATCH --requeue
#SBATCH --nodelist=aimi-compute-01,aimi-compute-02,aimi-compute-03


# Load the required environment
source activate mem

# Navigate to the working directory
cd /home/lz586/icl/Multi-Hop-Reasoning/


DATASET=hotpotqa-v2

# Same 3B SFT checkpoint the production GRPO is using.
MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen2.5-3B-SFT_hotpotqa_ep3_bsz48_lr5e-5_th-1

RETRIEVAL_THRESHOLD=0.6
TOP_K=4
NUM_GENERATIONS=32
NUM_DB_ROLLOUTS=4

DATABASE_PATH=""
THRESHOLD=-1
REWARD_FUNC="f1"
TWO_PHASE=""
DEBUG=""

export TRITON_CACHE_DIR=/home/lz586/.cache/triton
export TMPDIR=/home/lz586/.tmp
mkdir -p $TRITON_CACHE_DIR $TMPDIR

# ── THE PROBE: force FSDP for a model branch (3B) that defaults to DDP ────────
export FORCE_FSDP=1

if [ "${DATASET}" = "hotpotqa-v2" ]; then
    TRAIN_SIZE=7000
    TOTAL_BATCH_SIZE=512
    DATASET_NAME="hotpotqa"
    TWO_PHASE="--two_phase"
else
    echo "This probe only supports hotpotqa-v2"; exit 1
fi

USE_CHAT_TEMPLATE=""
if [[ "${MODEL_PATH}" == *"Qwen/Qwen"* ]]; then
    PHASE1_PROMPT_TYPE="formatted_zero_rl_v5"
    USE_CHAT_TEMPLATE="--use_chat_template"
else
    PHASE1_PROMPT_TYPE="sft"
fi

echo "Starting 3B-with-FSDP probe..."
bash /home/lz586/icl/Multi-Hop-Reasoning/scripts/grpo_train.sh \
--gpu_type B200 \
--model_path ${MODEL_PATH} \
--dataset_name ${DATASET_NAME} \
--database_path "${DATABASE_PATH}" \
--train_size ${TRAIN_SIZE} \
--reward_func ${REWARD_FUNC} \
--total_batch_size ${TOTAL_BATCH_SIZE} \
--phase1_prompt_type ${PHASE1_PROMPT_TYPE} \
${USE_CHAT_TEMPLATE} \
${TWO_PHASE} \
${DEBUG} \
--retrieval_threshold ${RETRIEVAL_THRESHOLD} \
--top_k ${TOP_K} \
--num_generations ${NUM_GENERATIONS} \
--num_db_rollouts ${NUM_DB_ROLLOUTS}

echo "Probe complete."
