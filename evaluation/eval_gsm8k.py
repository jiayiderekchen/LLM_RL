"""
Evaluate a (PPO-trained or base) checkpoint on GSM8K test set.

Usage:
  python evaluation/eval_gsm8k.py \
      --model_path checkpoints/qwen3_gsm8k_ppo/global_step_500 \
      --split test \
      --max_samples 1319

Outputs a JSON accuracy report to evaluation/results/.
"""

import argparse
import json
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from rewards.gsm8k_reward import compute_score

SYSTEM_PROMPT = "You are a helpful assistant that solves math problems step by step."


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def evaluate(
    model_path: str,
    split: str = "test",
    max_samples: int = -1,
    max_new_tokens: int = 512,
    batch_size: int = 8,
    device: str = "cuda",
) -> tuple:
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched generation

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.")

    ds = load_dataset("gsm8k", "main")[split]
    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    correct = 0
    total = 0
    results = []
    t0 = time.time()

    # Build all prompts upfront
    all_prompts = [build_prompt(tokenizer, s["question"]) for s in ds]
    all_answers = [s["answer"] for s in ds]
    all_questions = [s["question"] for s in ds]

    pbar = tqdm(total=len(all_prompts), desc="Evaluating", unit="sample")

    for batch_start in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[batch_start: batch_start + batch_size]
        batch_answers = all_answers[batch_start: batch_start + batch_size]
        batch_questions = all_questions[batch_start: batch_start + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Decode only new tokens per sample.
        # With left-padding, all rows in the batch share the same padded input
        # length (inputs["input_ids"].shape[1]), so we slice uniformly.
        input_len = inputs["input_ids"].shape[1]
        for j in range(len(batch_prompts)):
            gen_ids = output_ids[j][input_len:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)
            score = compute_score("gsm8k", response, batch_answers[j])
            correct += int(score == 1.0)
            total += 1
            results.append(
                {
                    "question": batch_questions[j],
                    "ground_truth": batch_answers[j],
                    "response": response,
                    "correct": score == 1.0,
                }
            )

        pbar.update(len(batch_prompts))
        pbar.set_postfix(accuracy=f"{correct/total:.3f}", samples_per_s=f"{total/(time.time()-t0):.1f}")

    pbar.close()

    accuracy = correct / total
    report = {
        "model_path": model_path,
        "split": split,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
    }
    print(f"\nFinal GSM8K accuracy: {accuracy:.4f} ({correct}/{total})")
    return report, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    # If a verl checkpoint dir is passed, auto-resolve to actor/huggingface subdir
    # or instruct user to run model_merger.py first.
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", default="evaluation/results")
    args = parser.parse_args()

    model_path = args.model_path
    # Auto-resolve verl checkpoint structure:
    # global_step_N/actor/huggingface  (if already merged to HF format)
    for candidate in [
        os.path.join(model_path, "actor", "huggingface"),
        os.path.join(model_path, "actor"),
    ]:
        if os.path.isfile(os.path.join(candidate, "config.json")):
            print(f"[eval] Resolved verl checkpoint → {candidate}")
            model_path = candidate
            break

    if not os.path.isfile(os.path.join(model_path, "config.json")) and not model_path.startswith("Qwen/"):
        print(
            f"ERROR: No config.json found in {model_path}\n"
            "This is an FSDP shard checkpoint. Convert it first:\n\n"
            "  python /content/verl/scripts/model_merger.py \\\n"
            "      --backend fsdp \\\n"
            f"      --hf_model_path Qwen/Qwen3-1.7B \\\n"
            f"      --local_dir {args.model_path}/actor \\\n"
            f"      --output_dir {args.model_path}_hf\n\n"
            f"Then eval with --model_path {args.model_path}_hf"
        )
        sys.exit(1)

    report, results = evaluate(
        model_path,
        split=args.split,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(model_path.rstrip("/"))
    out_path = os.path.join(args.output_dir, f"{model_name}_results.json")
    with open(out_path, "w") as f:
        json.dump({"report": report, "samples": results}, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
