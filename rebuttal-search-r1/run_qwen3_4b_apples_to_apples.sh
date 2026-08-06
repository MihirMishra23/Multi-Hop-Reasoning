#!/usr/bin/env bash
# Backward-compatible entrypoint used by Linxi's Qwen3-4B jobs.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
export MODEL_RUN_NAME="${MODEL_RUN_NAME:-qwen3-4b}"
exec "${SCRIPT_DIR}/run_qwen3_training.sh" "$@"
