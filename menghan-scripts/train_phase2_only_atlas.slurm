#!/bin/bash
# Experiment B Step 2: Train Phase-2-only GRPO with atlas-rag KB.
#
# Mirrors scripts/lx_scripts/grpo_training_1phase.slurm 1:1, EXCEPT:
#   DATABASE_PATH → atlas-rag KB (built from same hotpotqa train [82347, 90447) slice)
# All other hyperparams (TRAIN_SIZE, TOTAL_BATCH_SIZE, RETRIEVAL_THRESHOLD,
# TOP_K, PHASE1_PROMPT_TYPE, MODEL_PATH=1phase SFT base) are identical.
#
# Produces a Phase-2-only GRPO ckpt that can be compared apples-to-apples
# with the Gemini Phase-2-only baseline (40.7/17.9/55.0 in Table 5).
#
# Prereqs:
#   - kb_baselines/autoschemakg/output/kbs/hotpotqa_train_82347_8100_seed42_kb.json
#     exists (run menghan-scripts/build_atlas_kb_train.slurm first)
#
# Usage:
#   bash menghan-scripts/train_phase2_only_atlas.slurm

#SBATCH --job-name=phase2only-atlas-1.7b-4gpu
#SBATCH --output=/home/mx253/slurm_output/lmlm_multihop/phase2only_atlas_1.7b_4gpu_%j.out
#SBATCH --error=/home/mx253/slurm_output/lmlm_multihop/phase2only_atlas_1.7b_4gpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=256G
#SBATCH --time=12:00:00
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
PHASE1_PROMPT_TYPE="sft"       # auto-set when MODEL_PATH is non-Qwen-base SFT
# 1phase SFT base — same as Linxi used
MODEL_PATH=/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_1phase

# atlas-rag KB built from SAME train slice as Gemini KB (start=82347, n=8100)
DATABASE_PATH=/share/j_sun/mx253/Multi-Hop-Reasoning/kb_baselines/autoschemakg/output/kbs/hotpotqa_train_82347_8100_seed42_kb.json

# Save under atlas-specific dir to AVOID overwriting Linxi's Gemini 1phase ckpt
# (which has the EXACT SAME output_dir name under default SAVE_DIR=.../checkpoints/main/)
SAVE_DIR=/share/j_sun/lmlm_multihop/checkpoints/atlas_rag

if [[ ! -f "${DATABASE_PATH}" ]]; then
    echo "ERROR: atlas-rag train KB not found at ${DATABASE_PATH}"
    echo "Run: bash menghan-scripts/build_atlas_kb_train.slurm  first."
    exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: 1phase SFT base model not found at ${MODEL_PATH}"
    exit 1
fi

echo "============================================================"
echo "[GRPO Phase-2-only training: atlas-rag KB]"
echo "  SFT base:  ${MODEL_PATH##*/}"
echo "  KB:        ${DATABASE_PATH}"
echo "  Slice:     hotpotqa train, seed=42, [82347, 90447)"
echo "  Train size:${TRAIN_SIZE}"
echo "  Reward:    ${REWARD_FUNC}"
echo "  --two_phase: OFF (single-phase mode)"
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
    # NOTE: --use_inverses NOT passed — matches Linxi's 1phase script (also omits it)

echo
echo "==== GRPO Phase-2-only training DONE ===="
echo "Look for ckpt under /share/j_sun/lmlm_multihop/checkpoints/main/ with the"
echo "model dir name printed by grpo_train.sh above."
