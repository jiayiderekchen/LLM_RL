"""
Convert a verl FSDP single-GPU checkpoint to HuggingFace format.

Usage:
    python scripts/convert_checkpoint.py \
        --ckpt_dir /content/drive/MyDrive/LLM_RL_runs/checkpoints/global_step_870 \
        --base_model Qwen/Qwen3-1.7B \
        --output_dir /content/drive/MyDrive/LLM_RL_runs/checkpoints/global_step_870_hf
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def convert(ckpt_dir: str, base_model: str, output_dir: str) -> None:
    shard = os.path.join(ckpt_dir, "actor", "model_world_size_1_rank_0.pt")
    hf_cfg_dir = os.path.join(ckpt_dir, "actor", "huggingface")

    if not os.path.exists(shard):
        raise FileNotFoundError(f"FSDP shard not found: {shard}")

    print(f"Loading base model architecture from {base_model} ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"Loading FSDP state dict from {shard} ...")
    state = torch.load(shard, map_location="cpu", weights_only=False)

    # verl wraps the model in FSDP; unwrap common prefixes
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    # Strip common FSDP / torch.compile prefixes
    cleaned = {}
    for k, v in state.items():
        key = k
        for prefix in ("_orig_mod.", "module.", "_fsdp_wrapped_module.",
                       "base_model.model.", "_checkpoint_wrapped_module."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  Missing keys  ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
    print("State dict loaded.")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving HF model to {output_dir} ...")
    model.save_pretrained(output_dir)

    # Copy tokenizer from the checkpoint's huggingface dir (has chat template etc.)
    src_tok = hf_cfg_dir if os.path.exists(hf_cfg_dir) else base_model
    tokenizer = AutoTokenizer.from_pretrained(src_tok, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    print(f"\nDone. HF model saved to: {output_dir}")
    print("Run eval with:")
    print(f"  python evaluation/eval_gsm8k.py --model_path {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True,
                        help="Path to global_step_N directory")
    parser.add_argument("--base_model", default="Qwen/Qwen3-1.7B",
                        help="HF model id for architecture + tokenizer")
    parser.add_argument("--output_dir", required=True,
                        help="Where to save the converted HF model")
    args = parser.parse_args()
    convert(args.ckpt_dir, args.base_model, args.output_dir)
