#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --partition=jjs533
#SBATCH --constraint=gpu-high
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/rtn27/popqa1399_%x_%j.out
#SBATCH --error=/home/rtn27/popqa1399_%x_%j.err

set -euo pipefail

: "${METHOD:?Set METHOD}"
: "${MODEL_KEY:?Set MODEL_KEY}"
: "${RUN_TAG:?Set RUN_TAG}"

REPO_DIR="${REPO_DIR:-/home/rtn27/Multi-Hop-Reasoning-popqa-longtail}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/share/j_sun/rtn27/popqa_longtail_1399}"
OUTPUT_DIR="${OUTPUT_DIR:-/share/j_sun/rtn27/popqa_longtail_eval/${RUN_TAG}}"
CORPUS_PATH="${ARTIFACT_DIR}/popqa_longtail_contexts.jsonl"
CACHE_ROOT="${CACHE_ROOT:-/scratch/${USER}/eval_cache}"
JOB_TMP="${JOB_TMP:-/scratch/${USER}/eval_tmp/${SLURM_JOB_ID}}"
EVAL_ENV="${EVAL_ENV:-/scratch/rtn27/multi-hop-reasoning}"

mkdir -p \
    "${CACHE_ROOT}/huggingface" \
    "${CACHE_ROOT}/torch" \
    "${CACHE_ROOT}/vllm" \
    "${JOB_TMP}/triton"
export TMPDIR="${JOB_TMP}"
export TMP="${JOB_TMP}"
export TEMP="${JOB_TMP}"
export HF_HOME="${CACHE_ROOT}/huggingface"
export TORCH_HOME="${CACHE_ROOT}/torch"
export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
export TRITON_CACHE_DIR="${JOB_TMP}/triton"

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
# The lz586 ``mem`` environment's MKL activation hook reads an unset variable.
# Temporarily disable nounset only for Conda activation, then restore it.
set +u
conda activate "${EVAL_ENV}"
set -u
cd "${REPO_DIR}"
export POPQA_CORPUS_PATH="${CORPUS_PATH}"

case "${MODEL_KEY}" in
    qwen3-1.7b)
        MODEL="qwen3-1.7b"
        MODEL_PATH=""
        ;;
    qwen3-4b)
        MODEL="qwen3-4b"
        MODEL_PATH=""
        ;;
    kbevo-1.7b)
        MODEL=""
        MODEL_PATH="/share/j_sun/lmlm_multihop/checkpoints/main/Qwen3-1.7B-SFT_hotpotqa_ep3_bsz48_th-1_2phase_classic_retrieval_6k-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak/checkpoint-500"
        ;;
    kbevo-4b)
        MODEL=""
        MODEL_PATH="/share/j_sun/lmlm_multihop/checkpoints/main/Qwen3-4B-SFT_hotpotqa_ep3_bsz48_th-1-grpo-tbs512-N32-K4-B16-M7-b0.0-lr5e-6-step500-n7000-f1-2ph-prsft-wcount-th0.6-topk4-nak/checkpoint-500"
        ;;
    *)
        echo "Unknown MODEL_KEY=${MODEL_KEY}" >&2
        exit 2
        ;;
esac

COMMON_ARGS=(
    --dataset popqa
    --dataset-source hf
    --setting long_tail
    --split test
    --seed 42
    --start-index 0
    --total-count 1399
    --max-tokens 1024
    --save-every-examples 64
    --output-dir "${OUTPUT_DIR}"
    --save-version "_${RUN_TAG}"
    --eval
)

echo "code_commit=$(git rev-parse HEAD)"
echo "method=${METHOD} model_key=${MODEL_KEY} run_tag=${RUN_TAG}"
echo "manifest=${ARTIFACT_DIR}/manifest.json corpus=${CORPUS_PATH}"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader

if [[ "${METHOD}" == "two_phase" ]]; then
    python src/eval_multihop.py \
        --method two_phase \
        --model-path "${MODEL_PATH}" \
        --batch-size 64 \
        --top-k 4 \
        --similarity-threshold 0.6 \
        --use-inverses \
        --use-train-params \
        --concat-all-db \
        --use-contexts all \
        "${COMMON_ARGS[@]}"
elif [[ "${METHOD}" == "rag" ]]; then
    python src/eval_multihop.py \
        --method rag \
        --model "${MODEL}" \
        --batch-size 1 \
        --rag-k 4 \
        --rag-corpus-path "${CORPUS_PATH}" \
        "${COMMON_ARGS[@]}"
elif [[ "${METHOD}" == "direct" ]]; then
    python src/eval_multihop.py \
        --method direct \
        --model "${MODEL}" \
        --batch-size 32 \
        "${COMMON_ARGS[@]}"
else
    echo "Unsupported METHOD=${METHOD}" >&2
    exit 2
fi
