#!/usr/bin/env bash
# Evaluate a checkpoint (or base model) on GSM8K test set.
#
# Usage:
#   bash scripts/evaluate.sh                                         # base model
#   bash scripts/evaluate.sh checkpoints/qwen3_gsm8k_ppo/global_step_500

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

MODEL_PATH=${1:-"Qwen/Qwen3-1.7B"}

echo "=== Evaluating: ${MODEL_PATH} ==="
python evaluation/eval_gsm8k.py \
    --model_path "${MODEL_PATH}" \
    --split test \
    --max_new_tokens 512

echo "Done. Results saved in evaluation/results/"
