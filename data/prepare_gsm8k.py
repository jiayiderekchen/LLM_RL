"""
Prepare GSM8K dataset for VERL PPO training.

Outputs:
  data/gsm8k/train.parquet
  data/gsm8k/test.parquet

Each row has columns:
  prompt  - formatted chat prompt (list of dicts, OpenAI message format)
  answer  - ground-truth answer string containing "#### <number>"
"""

import argparse
import os

import pandas as pd
from datasets import load_dataset


SYSTEM_PROMPT = "You are a helpful assistant that solves math problems step by step."


def format_prompt(question: str) -> list[dict]:
    """Build the chat prompt that will be fed to the tokenizer."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def prepare(output_dir: str = "data/gsm8k") -> None:
    os.makedirs(output_dir, exist_ok=True)

    ds = load_dataset("gsm8k", "main")

    for split in ("train", "test"):
        rows = []
        for sample in ds[split]:
            rows.append(
                {
                    "prompt": format_prompt(sample["question"]),
                    "answer": sample["answer"],
                    "data_source": "gsm8k",
                    "reward_model": {
                        "ground_truth": sample["answer"],
                        "style": "rule",
                    },
                }
            )

        df = pd.DataFrame(rows)
        out_path = os.path.join(output_dir, f"{split}.parquet")
        df.to_parquet(out_path, index=False)
        print(f"Saved {len(df)} {split} samples → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/gsm8k")
    args = parser.parse_args()
    prepare(args.output_dir)
