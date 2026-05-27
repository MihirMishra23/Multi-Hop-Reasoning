#!/bin/bash
# Experiment B sanity check: Rerun Linxi's Phase-2-only Gemini training, but
# with the new trainer code (retrieval_top_k=1 instead of the old buggy top_k=4
# load_database default). This produces the apples-to-apples "Gemini DB" row
# at retrieval_top_k=1 — directly comparable to our atlas-rag training run.
#
# Mirrors train_phase2_only_atlas.slurm 1:1, EXCEPT:
#   DATABASE_PATH → Gemini train KB (same slice as atlas, hotpotqa train [82347, 90447))
#   SAVE_DIR      → /share/j_sun/lmlm_multihop/checkpoints/gemini_rerun_top1  (don't collide)
#   JOB_NAME      → distinct
#
# All other hyperparams (TRAIN_SIZE, TOTAL_BATCH_SIZE, RETRIEVAL_THRESHOLD,
# TOP_K, PHASE1_PROMPT_TYPE, MODEL_PATH=1phase SFT base, no --use_inverses)
# are identical to train_phase2_only_atlas.slurm and to Linxi's original
# 1phase script.
#
# Usage:
#   sbatch menghan-scripts/train_phase2_only_gemini_rerun.slurm

#SBATCH --job-name=phase2only-gemini-rerun-1.7b-4gpu
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/phase2only_gemini_rerun_1.7b_4gpu_%j.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/phase2only_gemini_rerun_1.7b_4gpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --partition=aimi,default_partition
#SBATCH --gres=gpu:nvidia_b200:4
#SBATCH --exclude=jjs533-compute-01,jjs533-compute-02,jjs533-compute-03
#SBATCH --requeue

set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate lmlm_b200
cd /share/j_sun/mx253/Multi-Hop-Reasoning

# ── Config (mirrors grpo_training_1phase.slurm "hotpotqa-v2-onlyphase1") ─────
DATASET_NAME=hotpotqa
TRAIN_SIZE=7000
TOTAL_BATCH_SIZE=512
RETRIEVAL_THRESHOLD=0.6
TOP_K=4
REWARD_FUNC="f1"               # Linxi's 1phase script default
PHASE1_PROMPT_TYPE="sft"
# 1phase SFT base — same as Linxi used
MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_1phase

# Gemini train KB — same slice as atlas (hotpotqa train [82347, 90447))
DATABASE_PATH=/share/j_sun/lmlm_multihop/database/gemini/hotpotqa_train_start_idx_82347_nb_8100_database.json

# Save under gemini_rerun-specific dir to avoid colliding with Linxi's debug/main ckpts
SAVE_DIR=/share/j_sun/lmlm_multihop/checkpoints/gemini_rerun_top1

if [[ ! -f "${DATABASE_PATH}" ]]; then
    echo "ERROR: Gemini train KB not found at ${DATABASE_PATH}"
    exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: 1phase SFT base model not found at ${MODEL_PATH}"
    exit 1
fi

echo "============================================================"
echo "[GRPO Phase-2-only training: Gemini KB rerun w/ retrieval_top_k=1]"
echo "  SFT base:  ${MODEL_PATH##*/}"
echo "  KB:        ${DATABASE_PATH}"
echo "  Slice:     hotpotqa train, seed=42, [82347, 90447)"
echo "  Train size:${TRAIN_SIZE}"
echo "  Reward:    ${REWARD_FUNC}"
echo "  --two_phase: OFF (single-phase mode)"
echo "  retrieval_top_k: 1 (via modified trainer load_database call)"
echo "============================================================"

mkdir -p "${SAVE_DIR}"

bash scripts/grpo_train.sh \
    --gpu_type B200 \
    --model_path "${MODEL_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --database_path "${DATABASE_PATH}" \
    --save_dir "${SAVE_DIR}" \
    --train_size ${TRAIN_SIZE} \
    --reward_func ${REWARD_FUNC} \
    --total_batch_size ${TOTAL_BATCH_SIZE} \
    --phase1_prompt_type ${PHASE1_PROMPT_TYPE} \
    --retrieval_threshold ${RETRIEVAL_THRESHOLD} \
    --top_k ${TOP_K} \
    --save_steps 50
    # NOTE: --two_phase intentionally omitted (single-phase mode)
    # NOTE: --use_inverses NOT passed — matches Linxi's 1phase script

echo
echo "==== GRPO Phase-2-only Gemini rerun DONE ===="
echo "Final ckpt: ${SAVE_DIR}/Qwen3-1.7B-SFT_..._1phase-grpo-...-nak/checkpoint-500"
