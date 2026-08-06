#!/bin/bash
# One reproducible Table 3 evaluation job. Configure with sbatch --export.
# Required: METHOD, MODEL_KEY, DATASET, RUN_TAG
# Optional: NUM_SAMPLES (default 1000), OUTPUT_DIR
#
# MODEL_KEY values:
#   qwen3-1.7b, qwen3-4b
#   kbevo-sft-1.7b, kbevo-grpo-1.7b, kbevo-sft-4b, kbevo-grpo-4b

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --partition=default_partition
#SBATCH --nodelist=unicorn-compute-[01-02]
#SBATCH --gres=gpu:nvidia_a40:1
#SBATCH --output=/home/rtn27/table3_%x_%j.out
#SBATCH --error=/home/rtn27/table3_%x_%j.err

set -euo pipefail

: "${METHOD:?Set METHOD}"
: "${MODEL_KEY:?Set MODEL_KEY}"
: "${DATASET:?Set DATASET}"
: "${RUN_TAG:?Set RUN_TAG}"

NUM_SAMPLES="${NUM_SAMPLES:-1000}"
OUTPUT_DIR="${OUTPUT_DIR:-/share/j_sun/rtn27/table3_reproduction}"
REPO_DIR="${REPO_DIR:-/home/rtn27/Multi-Hop-Reasoning-ircot-table3}"

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate multi-hop-reasoning
cd "${REPO_DIR}"

case "${MODEL_KEY}" in
    qwen3-1.7b)
        LLM_MODEL="qwen3-1.7b"
        MODEL_PATH="unused"
        ;;
    qwen3-4b)
        LLM_MODEL="qwen3-4b"
        MODEL_PATH="unused"
        ;;
    kbevo-sft-1.7b)
        LLM_MODEL="unused"
        MODEL_PATH="/share/j_sun/lmlm_multihop/checkpoints/main/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k_ret_0.6_top_k_4"
        ;;
    kbevo-grpo-1.7b)
        LLM_MODEL="unused"
        MODEL_PATH="/share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak/checkpoint-500"
        ;;
    kbevo-sft-4b)
        LLM_MODEL="unused"
        MODEL_PATH="/share/j_sun/lz586/checkpoints/lmlm_multi_hop/Qwen3-4B-SFT_hotpotqa_ep3_bsz48_th-1"
        ;;
    kbevo-grpo-4b)
        LLM_MODEL="unused"
        MODEL_PATH="/share/j_sun/lmlm_multihop/checkpoints/debug/Qwen3-4B-SFT_hotpotqa_ep3_bsz48_th-1-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak/checkpoint-500"
        ;;
    *)
        echo "Unknown MODEL_KEY=${MODEL_KEY}" >&2
        exit 2
        ;;
esac

if [[ "${METHOD}" == "two_phase" ]]; then
    case "${MODEL_KEY}" in
        kbevo-*) ;;
        *) echo "two_phase requires a kbevo-* MODEL_KEY" >&2; exit 2 ;;
    esac
    EXTRA_ARGS=(--use-train-params --concat-all-db --use-contexts all)
else
    case "${MODEL_KEY}" in
        qwen3-*) ;;
        *) echo "${METHOD} requires a qwen3-* MODEL_KEY" >&2; exit 2 ;;
    esac
    EXTRA_ARGS=()
fi

if [[ "${METHOD}" == "ircot" ]]; then
    : "${IRCOT_RETRIEVER_URL:?Set IRCOT_RETRIEVER_URL to the official retrieval service}"
    IRCOT_EXTRA_ARGS=(
        --ircot-retriever-url "${IRCOT_RETRIEVER_URL}"
        --ircot-prompt-dir "${IRCOT_PROMPT_DIR:-provenance/ircot/prompts}"
    )
else
    IRCOT_EXTRA_ARGS=()
fi

echo "code_commit=$(git rev-parse HEAD)"
echo "method=${METHOD} model_key=${MODEL_KEY} dataset=${DATASET} num_samples=${NUM_SAMPLES}"
echo "model_path=${MODEL_PATH} llm_model=${LLM_MODEL} run_tag=${RUN_TAG}"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader

bash scripts/eval_lmlm_multihop.sh \
    --model_path "${MODEL_PATH}" \
    --method "${METHOD}" \
    --llm_model "${LLM_MODEL}" \
    --dataset "${DATASET}" \
    --split dev \
    --num_samples "${NUM_SAMPLES}" \
    --output-dir "${OUTPUT_DIR}" \
    --save_version "${RUN_TAG}" \
    --top-k 4 \
    --similarity-threshold 0.6 \
    --embedding-batch-size "${EMBEDDING_BATCH_SIZE:-2048}" \
    --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.6}" \
    --ircot-retrieval-k 6 \
    --ircot-max-evidence 15 \
    --ircot-max-steps 10 \
    --ircot-generator-max-tokens 300 \
    --ircot-distractor-count "${IRCOT_DISTRACTOR_COUNT:-2}" \
    "${IRCOT_EXTRA_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
