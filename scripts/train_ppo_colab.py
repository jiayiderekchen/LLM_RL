"""
VERL PPO training launcher for Colab terminal.

Uses Hydra's compose API so all of verl's config defaults are loaded properly,
then calls run_ppo() directly (bypassing the @hydra.main decorator).

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

    sys.path.insert(0, args.repo_dir)

    import torch
    import verl

    verl_cfg_dir = os.path.join(
        os.path.dirname(os.path.abspath(verl.__file__)), "trainer/config"
    )

    gpu_mem_gb = (
        torch.cuda.get_device_properties(0).total_memory / 1e9
        if torch.cuda.is_available() else 0
    )
    is_a100 = gpu_mem_gb >= 38
    print(f"GPU memory: {gpu_mem_gb:.1f} GB  →  {'A100' if is_a100 else 'T4'} config")

    train_files = os.path.join(args.repo_dir, "data/gsm8k/train.parquet")
    val_files   = os.path.join(args.repo_dir, "data/gsm8k/test.parquet")
    reward_path = os.path.join(args.repo_dir, "rewards/gsm8k_reward.py")
    ckpt_dir    = os.path.join(args.repo_dir, "checkpoints")
    logger      = "wandb" if args.use_wandb else "console"

    # ── Base overrides (same as train_ppo.sh) ──────────────────────────────
    overrides = [
        f"data.train_files={train_files}",
        f"data.val_files={val_files}",
        "data.max_prompt_length=512",
        "data.max_response_length=1024",
        "data.train_batch_size=128",
        "data.return_raw_chat=true",
        "actor_rollout_ref.model.path=Qwen/Qwen3-1.7B",
        "actor_rollout_ref.model.use_remove_padding=false",
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        "actor_rollout_ref.actor.ppo_mini_batch_size=64",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.actor.entropy_coeff=0.001",
        "actor_rollout_ref.actor.use_kl_loss=false",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.optim.lr_warmup_steps=10",
        "actor_rollout_ref.actor.optim.min_lr_ratio=0.0",
        "actor_rollout_ref.actor.optim.warmup_style=constant",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.4",
        "actor_rollout_ref.rollout.ignore_eos=false",
        "actor_rollout_ref.rollout.do_sample=true",
        "actor_rollout_ref.ref.fsdp_config.param_offload=true",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
        "critic.model.path=Qwen/Qwen3-1.7B",
        "critic.model.tokenizer_path=Qwen/Qwen3-1.7B",
        "critic.model.use_remove_padding=false",
        "critic.model.enable_gradient_checkpointing=true",
        "critic.optim.lr=2e-6",
        "critic.optim.lr_warmup_steps=10",
        "critic.optim.min_lr_ratio=0.0",
        "critic.optim.warmup_style=constant",
        "critic.ppo_micro_batch_size_per_gpu=4",
        f"reward.custom_reward_function.path={reward_path}",
        "reward.custom_reward_function.name=compute_score",
        f"custom_reward_function.path={reward_path}",
        "custom_reward_function.name=compute_score",
        "algorithm.adv_estimator=gae",
        "algorithm.use_kl_in_reward=false",
        "algorithm.kl_ctrl.kl_coef=0.001",
        "trainer.balance_batch=true",
        "trainer.total_epochs=15",
        "trainer.project_name=gsm8k_ppo",
        f"trainer.experiment_name={args.experiment_name}",
        f"trainer.logger=[{logger}]",
        "trainer.log_val_generations=0",
        "trainer.nnodes=1",
        f"trainer.n_gpus_per_node={args.n_gpus}",
        "trainer.save_freq=50",
        "trainer.test_freq=25",
        f"trainer.default_local_dir={ckpt_dir}",
    ]

    # ── T4-specific memory reductions ───────────────────────────────────────
    if not is_a100:
        overrides += [
            "data.train_batch_size=32",
            "data.max_prompt_length=384",
            "data.max_response_length=512",
            "actor_rollout_ref.actor.ppo_mini_batch_size=16",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
            "actor_rollout_ref.rollout.gpu_memory_utilization=0.35",
            "actor_rollout_ref.actor.fsdp_config.param_offload=true",
            "critic.ppo_micro_batch_size_per_gpu=2",
            "trainer.total_epochs=10",
            "trainer.test_freq=50",
        ]

    # ── Compose config from verl's defaults + our overrides ─────────────────
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=verl_cfg_dir, version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)

    from omegaconf import OmegaConf
    print("Final config:")
    print(OmegaConf.to_yaml(config))

    # ── Run training (bypass @hydra.main decorator) ──────────────────────────
    from verl.trainer.main_ppo import run_ppo
    run_ppo(config)


if __name__ == "__main__":
    main()
