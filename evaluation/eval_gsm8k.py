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
    device: str = "cuda",
) -> dict:
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    ds = load_dataset("gsm8k", "main")[split]
    if max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    correct = 0
    total = 0
    results = []
    t0 = time.time()

    for i, sample in enumerate(ds):
        prompt_text = build_prompt(tokenizer, sample["question"])
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy decoding for eval
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        # Decode only the new tokens
        gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True)

        score = compute_score(response, sample["answer"])
        correct += int(score == 1.0)
        total += 1

        results.append(
            {
                "question": sample["question"],
                "ground_truth": sample["answer"],
                "response": response,
                "correct": score == 1.0,
            }
        )

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(
                f"  [{i+1}/{len(ds)}] accuracy={correct/total:.3f}  "
                f"elapsed={elapsed:.0f}s"
            )

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
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_dir", default="evaluation/results")
    args = parser.parse_args()

    report, results = evaluate(
        args.model_path,
        split=args.split,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(args.model_path.rstrip("/"))
    out_path = os.path.join(args.output_dir, f"{model_name}_results.json")
    with open(out_path, "w") as f:
        json.dump({"report": report, "samples": results}, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
