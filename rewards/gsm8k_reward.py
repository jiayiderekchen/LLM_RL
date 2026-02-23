"""
GSM8K verifiable correctness reward for VERL PPO training.

Reward signal:
  +1.0  if extracted final number matches ground truth
   0.0  otherwise

VERL calls compute_score(solution_str, ground_truth) during rollout scoring.
"""

import re
from typing import Optional


def _normalize_number(text: str) -> Optional[str]:
    """Strip formatting from a number string for comparison."""
    text = text.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        val = float(text)
        # Represent as int string if it's a whole number, else keep float
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return None


def extract_answer(text: str) -> Optional[str]:
    """
    Extract the final numeric answer from model output.

    Strategy:
    1. Look for GSM8K-style boxed answer: #### <number>
    2. Fall back to last number in the text.
    """
    # GSM8K gold answers use #### format
    boxed = re.search(r"####\s*(-?\d[\d,\.]*)", text)
    if boxed:
        return _normalize_number(boxed.group(1))

    # Model outputs: look for "= <number>" near end, then last number overall
    eq_match = re.findall(r"=\s*(-?\d[\d,\.]*)", text)
    if eq_match:
        return _normalize_number(eq_match[-1])

    numbers = re.findall(r"-?\d[\d,\.]*", text)
    if numbers:
        return _normalize_number(numbers[-1])

    return None


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """
    VERL reward interface.

    Args:
        data_source: Dataset name (e.g. 'gsm8k'). Unused here.
        solution_str: Full model-generated response text.
        ground_truth: Ground-truth answer string from GSM8K (contains #### <num>).
        extra_info: Optional extra metadata. Unused here.

    Returns:
        1.0 if correct, 0.0 otherwise.
    """
    pred = extract_answer(solution_str)
    gold = extract_answer(ground_truth)

    if pred is None or gold is None:
        return 0.0

    return 1.0 if pred == gold else 0.0


# --- unit tests (run with: python rewards/gsm8k_reward.py) ---
if __name__ == "__main__":
    cases = [
        ("The answer is 42.", "#### 42", 1.0),
        ("So the total is $1,234.", "#### 1234", 1.0),
        ("I think it's 100.", "#### 99", 0.0),
        ("No numbers here.", "#### 5", 0.0),
        ("Result = 3.5", "#### 3.5", 1.0),
        ("Result = 3.50", "#### 3.5", 1.0),
    ]
    passed = 0
    for sol, gt, expected in cases:
        got = compute_score("gsm8k", sol, gt)
        status = "PASS" if got == expected else "FAIL"
        if status == "PASS":
            passed += 1
        print(f"[{status}] sol={repr(sol[:40])} gt={repr(gt)} expected={expected} got={got}")
    print(f"\n{passed}/{len(cases)} tests passed")
