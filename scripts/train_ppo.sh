#!/usr/bin/env bash
# Launch VERL PPO training for Qwen3-1.7B on GSM8K.
# Adjust N_GPUS to match your hardware.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

N_GPUS=${N_GPUS:-2}
EXPERIMENT=${EXPERIMENT:-qwen3_1.7b_ppo}

echo "=== Preparing GSM8K dataset ==="
python data/prepare_gsm8k.py --output_dir data/gsm8k

echo "=== Starting PPO training with ${N_GPUS} GPU(s) ==="
python -m verl.trainer.main_ppo \
    --config-path "$(pwd)/configs" \
    --config-name qwen3_gsm8k_ppo \
    trainer.n_gpus_per_node="${N_GPUS}" \
    reward.custom_reward_function.path="$(pwd)/rewards/gsm8k_reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.experiment_name="${EXPERIMENT}" \
    "$@"
