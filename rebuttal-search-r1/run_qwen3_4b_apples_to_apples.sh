#!/usr/bin/env bash
# Backward-compatible entrypoint used by Linxi's Qwen3-4B jobs.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_ARTIFACT="${MODEL_ARTIFACT:-qwen3_4b}"
export MODEL_RUN_NAME="${MODEL_RUN_NAME:-qwen3-4b}"
exec "${SCRIPT_DIR}/run_qwen3_training.sh" "$@"
