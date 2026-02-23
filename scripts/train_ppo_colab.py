"""
VERL PPO training launcher for Colab.

Uses VERL's Python API to properly inject the custom GSM8K reward function,
avoiding the CLI Hydra struct conflict with reward_model.* keys.

Usage (from repo root):
    python scripts/train_ppo_colab.py \
        --repo_dir /content/LLM_RL \
        --n_gpus 1 \
        [--use_wandb]
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", default="/content/LLM_RL")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--experiment_name", default="qwen3_gsm8k_ppo_colab")
    return parser.parse_args()


def main():
    args = parse_args()

    # Add repo to path so rewards/ is importable
    sys.path.insert(0, args.repo_dir)

    import torch
    from omegaconf import OmegaConf

    # ── Load base config ────────────────────────────────────────────────────
    config_path = os.path.join(args.repo_dir, "configs", "qwen3_gsm8k_ppo.yaml")
    config = OmegaConf.load(config_path)

    # ── GPU-aware overrides ─────────────────────────────────────────────────
    gpu_mem_gb = (
        torch.cuda.get_device_properties(0).total_memory / 1e9
        if torch.cuda.is_available() else 0
    )
    is_a100 = gpu_mem_gb >= 38
    print(f"GPU memory: {gpu_mem_gb:.1f} GB  →  {'A100' if is_a100 else 'T4'} config")

    if not is_a100:
        config.data.train_batch_size = 32
        config.data.max_prompt_length = 384
        config.data.max_response_length = 512
        config.actor_rollout_ref.actor.ppo_mini_batch_size = 16
        config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 2
        config.actor_rollout_ref.rollout.response_length = 512
        config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.35
        config.actor_rollout_ref.model.enable_gradient_checkpointing = True
        OmegaConf.update(config, "actor_rollout_ref.model.fsdp_config.param_offload", True)
        config.critic.ppo_micro_batch_size_per_gpu = 2
        config.critic.model.enable_gradient_checkpointing = True
        OmegaConf.update(config, "critic.model.fsdp_config.param_offload", True)
        config.trainer.total_epochs = 10
        config.trainer.test_freq = 50

    # ── Path overrides ──────────────────────────────────────────────────────
    config.data.train_files = os.path.join(args.repo_dir, "data/gsm8k/train.parquet")
    config.data.val_files = os.path.join(args.repo_dir, "data/gsm8k/test.parquet")
    config.trainer.n_gpus_per_node = args.n_gpus
    config.trainer.default_local_dir = os.path.join(args.repo_dir, "checkpoints")
    config.trainer.experiment_name = args.experiment_name
    config.trainer.logger = ["console", "wandb"] if args.use_wandb else ["console"]

    # ── Reward function and compatibility keys ──────────────────────────────
    # Keep both legacy and new reward keys for cross-version VERL compatibility.
    OmegaConf.set_struct(config, False)
    config.custom_reward_function = OmegaConf.create({
        "path": os.path.join(args.repo_dir, "rewards/gsm8k_reward.py"),
        "name": "compute_score",
    })
    config.reward = OmegaConf.create({
        "custom_reward_function": {
            "path": os.path.join(args.repo_dir, "rewards/gsm8k_reward.py"),
            "name": "compute_score",
        }
    })
    # Newer VERL versions expect these top-level keys in struct mode.
    if "sandbox_fusion" not in config:
        config.sandbox_fusion = OmegaConf.create(
            {"enable": None, "url": None, "api_key": None, "timeout": None}
        )
    if "ray_kwargs" not in config:
        config.ray_kwargs = OmegaConf.create({})
    if "transfer_queue" not in config:
        config.transfer_queue = OmegaConf.create({"enable": False})
    OmegaConf.set_struct(config, True)

    print("Final config:")
    print(OmegaConf.to_yaml(config))

    # ── Launch VERL PPO trainer ─────────────────────────────────────────────
    from verl.trainer.main_ppo import main as verl_main
    verl_main(config)


if __name__ == "__main__":
    main()
