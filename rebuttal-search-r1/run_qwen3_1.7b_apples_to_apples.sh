#!/usr/bin/env bash
# Backward-compatible entrypoint used by Linxi's Qwen3-1.7B jobs.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_ARTIFACT="${MODEL_ARTIFACT:-qwen3_1_7b}"
export MODEL_RUN_NAME="${MODEL_RUN_NAME:-qwen3-1.7b}"
exec "${SCRIPT_DIR}/run_qwen3_training.sh" "$@"
