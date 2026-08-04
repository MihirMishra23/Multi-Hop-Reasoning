#!/bin/bash

# Rebuttal: 1.3B GRPO + Phase-1 grounded-COUNT reward (Route 2).
#
#   R_KB = avg_QA_F1 + lambda * min(num_grounded, cap) / cap
#
# Differs from the ratio-shaped grounding reward: this rewards adding
# correct triplets directly. Hallucinated triplets contribute 0 (no reward,
# no extra penalty). Cap saturates so a runaway KB cannot dominate.
#
# Submit:
#   sbatch --export=ALL,LAMBDA=0.5                scripts/lx_scripts/grpo_training_1.3b_grounded_count.slurm
#   sbatch --export=ALL,LAMBDA=1.0                scripts/lx_scripts/grpo_training_1.3b_grounded_count.slurm
#   sbatch --export=ALL,LAMBDA=0.5,GROUNDED_CAP=20 scripts/lx_scripts/grpo_training_1.3b_grounded_count.slurm

# SLURM Directives
#SBATCH -J grpo_1.3b_grdcnt
#SBATCH -o /home/lz586/icl/slurm_output/lmlm_multihop/grpo_1.3b_grdcnt_%j.out
#SBATCH -e /home/lz586/icl/slurm_output/lmlm_multihop/grpo_1.3b_grdcnt_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --get-user-env
#SBATCH --cpus-per-task=28
#SBATCH --mem=256G
#SBATCH -t infinite
#SBATCH --partition=aimi,kilian,jjs533,default_partition
#SBATCH --gres=gpu:nvidia_b200:1
#SBATCH --requeue
#SBATCH --nodelist=aimi-compute-01,aimi-compute-02,aimi-compute-03

source activate mem
cd /home/lz586/icl/Multi-Hop-Reasoning/

# ── Model & dataset ───────────────────────────────────────────────────────────
MODEL_PATH="${MODEL_PATH:-/share/j_sun/lmlm_multihop/checkpoints/main/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.6_top_k_4}"
DATASET_NAME="hotpotqa"
DATABASE_PATH=""

# ── Reward composition: R_KB = w_qa * F1 + lambda * min(num_grounded, cap)/cap
# Cap default 30 (matches typical 30-45 KB size; saturates at fully grounded).
# Override via env: GROUNDED_CAP=20 sbatch ...
LAMBDA="${LAMBDA:-0.5}"
W_QA=1.0
export DB_GROUNDED_CAP="${GROUNDED_CAP:-30}"
# reward_func order in src/grpo_train.py: f1 (col 0), grounded_count (col 1)
REWARD_FUNC="f1_grounded_count"
REWARD_WEIGHTS="${W_QA} ${LAMBDA}"

# ── Two-phase / generation knobs ──────────────────────────────────────────────
RETRIEVAL_THRESHOLD=0.6
TOP_K=4
NUM_GENERATIONS=32
NUM_DB_ROLLOUTS=4
TRAIN_SIZE=7000
TOTAL_BATCH_SIZE=512

# ── Prompt template ───────────────────────────────────────────────────────────
PHASE1_PROMPT_TYPE="sft"
USE_CHAT_TEMPLATE=""
if [[ "${MODEL_PATH}" == *"Qwen/Qwen"* ]]; then
    PHASE1_PROMPT_TYPE="formatted_zero_rl_v5"
    USE_CHAT_TEMPLATE="--use_chat_template"
fi

echo "Starting 1.3B grounded-count GRPO (lambda=${LAMBDA}, cap=${DB_GROUNDED_CAP})..."
echo "  REWARD_FUNC    = ${REWARD_FUNC}"
echo "  REWARD_WEIGHTS = ${REWARD_WEIGHTS}"
echo "  MODEL_PATH     = ${MODEL_PATH}"

bash /home/lz586/icl/Multi-Hop-Reasoning/scripts/grpo_train.sh \
    --gpu_type B200 \
    --model_path "${MODEL_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --database_path "${DATABASE_PATH}" \
    --train_size ${TRAIN_SIZE} \
    --total_batch_size ${TOTAL_BATCH_SIZE} \
    --reward_func "${REWARD_FUNC}" \
    --reward_weights "${REWARD_WEIGHTS}" \
    --phase1_prompt_type "${PHASE1_PROMPT_TYPE}" \
    ${USE_CHAT_TEMPLATE} \
    --two_phase \
    --retrieval_threshold ${RETRIEVAL_THRESHOLD} \
    --top_k ${TOP_K} \
    --num_generations ${NUM_GENERATIONS} \
    --num_db_rollouts ${NUM_DB_ROLLOUTS}

echo "1.3B grounded-count GRPO complete."
