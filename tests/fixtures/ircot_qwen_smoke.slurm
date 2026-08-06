#!/bin/bash
# End-to-end control-flow smoke test. The fixture endpoint is not BM25 and the
# resulting metrics are not reportable evaluation results.

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --partition=default_partition
#SBATCH --nodelist=unicorn-compute-03
#SBATCH --gres=gpu:1
#SBATCH --output=/home/rtn27/ircot_qwen_smoke_%j.out
#SBATCH --error=/home/rtn27/ircot_qwen_smoke_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/rtn27/Multi-Hop-Reasoning-ircot-table3}"
OUTPUT_DIR="${OUTPUT_DIR:-/share/j_sun/rtn27/table3_reproduction/ircot_method_smoke}"
FIXTURE_PORT="${FIXTURE_PORT:-18765}"

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate multi-hop-reasoning
cd "${REPO_DIR}"

python tests/fixtures/ircot_retriever_server.py --port "${FIXTURE_PORT}" &
fixture_pid=$!
trap 'kill "${fixture_pid}" 2>/dev/null || true' EXIT

python src/eval_multihop.py \
    --method ircot \
    --dataset hotpotqa \
    --setting distractor \
    --split dev \
    --model qwen3-1.7b \
    --batch-size 1 \
    --total-count 2 \
    --start-index 0 \
    --output-dir "${OUTPUT_DIR}" \
    --save-version "_fixture_${SLURM_JOB_ID}" \
    --save-every 1 \
    --temperature 0 \
    --max-tokens 300 \
    --ircot-retriever-url "http://127.0.0.1:${FIXTURE_PORT}" \
    --ircot-prompt-dir provenance/ircot/prompts \
    --ircot-retrieval-k 6 \
    --ircot-distractor-count 2 \
    --ircot-max-evidence 15 \
    --ircot-max-steps 10 \
    --ircot-generator-max-tokens 300 \
    --debug-evidence
