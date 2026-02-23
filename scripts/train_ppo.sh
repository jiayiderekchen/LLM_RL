#!/usr/bin/env bash
# Launch VERL PPO training for Qwen3-1.7B on GSM8K.
#
# Strategy: use verl's own ppo_trainer.yaml as the Hydra base config so all
# defaults are composed correctly.  Our project settings are passed as
# command-line overrides — no custom YAML is loaded as primary config.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

N_GPUS=${N_GPUS:-2}
EXPERIMENT=${EXPERIMENT:-qwen3_1.7b_ppo}

# Locate verl's Hydra config directory (works with pip install -e and normal install)
VERL_CFG=$(python3 -c "
import verl, os
print(os.path.join(os.path.dirname(os.path.abspath(verl.__file__)), 'trainer/config'))
")

echo "=== verl config dir: ${VERL_CFG} ==="

echo "=== Preparing GSM8K dataset ==="
python data/prepare_gsm8k.py --output_dir data/gsm8k

echo "=== Starting PPO training with ${N_GPUS} GPU(s) ==="
python -m verl.trainer.main_ppo \
    --config-path "${VERL_CFG}" \
    --config-name ppo_trainer \
    data.train_files="$(pwd)/data/gsm8k/train.parquet" \
    data.val_files="$(pwd)/data/gsm8k/test.parquet" \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.train_batch_size=128 \
    data.return_raw_chat=true \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.0 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.ignore_eos=false \
    actor_rollout_ref.rollout.do_sample=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    critic.model.path=Qwen/Qwen3-1.7B \
    critic.model.tokenizer_path=Qwen/Qwen3-1.7B \
    critic.model.use_remove_padding=false \
    critic.model.enable_gradient_checkpointing=true \
    critic.optim.lr=2e-6 \
    critic.optim.lr_warmup_steps=10 \
    critic.optim.min_lr_ratio=0.0 \
    critic.optim.warmup_style=constant \
    critic.ppo_micro_batch_size_per_gpu=4 \
    reward.custom_reward_function.path="$(pwd)/rewards/gsm8k_reward.py" \
    reward.custom_reward_function.name=compute_score \
    custom_reward_function.path="$(pwd)/rewards/gsm8k_reward.py" \
    custom_reward_function.name=compute_score \
    algorithm.adv_estimator=gae \
    algorithm.use_kl_in_reward=false \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.balance_batch=true \
    trainer.total_epochs=15 \
    trainer.project_name=gsm8k_ppo \
    trainer.experiment_name="${EXPERIMENT}" \
    trainer.logger=[console] \
    trainer.log_val_generations=0 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.default_local_dir="$(pwd)/checkpoints/qwen3_gsm8k_ppo" \
    "$@"
