# Qwen3-1.7B GSM8K PPO Replication (VERL)

## Objective

Replicate the reported improvement on GSM8K:

| Model             | GSM8K Accuracy |
| ----------------- | -------------- |
| Qwen3-1.7B (base) | ~69.2          |
| Qwen3-1.7B + PPO  | ~82.7          |

Method: **Reinforcement Learning with PPO using VERL** and a **verifiable correctness reward**.

This document specifies engineering requirements, architecture, and implementation steps so the project can be executed without needing to read the research paper in detail.

---

# 1. High-Level Overview

We will train Qwen3-1.7B using RL where:

1. Model generates reasoning solutions for GSM8K questions.
2. A program checks whether the final numeric answer is correct.
3. PPO reinforces trajectories that end with correct answers.
4. KL penalty prevents the model from drifting too far from the base model.

Pipeline:

```
GSM8K Question
      ↓
Rollout Generation (LLM sampling)
      ↓
Answer Extraction
      ↓
Automatic Verifier (Correct / Incorrect)
      ↓
Reward Signal
      ↓
PPO Update (Actor + Critic)
```

---

# 2. Tech Stack

## Core Framework

* VERL (https://github.com/verl-project/verl)
* PyTorch ≥ 2.2
* Transformers ≥ 4.44
* vLLM (for fast rollouts)

## Models

* Base model: `Qwen/Qwen3-1.7B`

## Dataset

* GSM8K (HuggingFace datasets)

## Hardware (recommended)

Minimum:

* 2× A100 40GB or equivalent

Paper setup:

* 4× H100 80GB

Lower hardware is possible by reducing batch sizes.

---

# 3. Repository Structure (Suggested)

```
project/
│
├── configs/
│   └── qwen3_gsm8k_ppo.yaml
│
├── rewards/
│   └── gsm8k_reward.py
│
├── data/
│   └── prepare_gsm8k.py
│
├── evaluation/
│   └── eval_gsm8k.py
│
└── scripts/
    ├── train_ppo.sh
    └── evaluate.sh
```

---

# 4. Environment Setup

```bash
git clone https://github.com/verl-project/verl
cd verl

pip install -e .
pip install datasets sympy regex
```

Install vLLM:

```bash
pip install vllm
```

Login to HuggingFace:

```bash
huggingface-cli login
```

---

# 5. Dataset Preparation

Load GSM8K:

```python
from datasets import load_dataset
ds = load_dataset("gsm8k", "main")
```

Each sample contains:

* `question`
* `answer` (contains final numeric result)

Store only training split for RL training.

---

# 6. Reward Function (CRITICAL)

Reward is **verifiable correctness**.

### Rules

* Extract final numeric value from model output.
* Normalize numbers:

  * remove commas
  * strip currency symbols
  * trim spaces
* Compare with ground truth.

### Reward

```
correct → +1.0
incorrect → 0.0
```

### Example

```python
def extract_answer(text):
    import re
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else None

def reward_fn(output, gt):
    pred = extract_answer(output)
    gold = extract_answer(gt)
    return 1.0 if pred == gold else 0.0
```

No strict formatting should be enforced.

---

# 7. PPO Training Configuration

Create:

`configs/qwen3_gsm8k_ppo.yaml`

Recommended starting parameters:

```
algorithm: PPO

actor_lr: 1e-6
critic_lr: 2e-6

train_batch_size: 128
minibatch_size: 64
microbatch_size: 16

ppo_clip_eps: 0.2
kl_beta: 1e-3

rollout_temperature: 1.0
max_response_length: 4096

training_steps: 500
```

Key ideas:

* Temperature 1.0 during rollout (exploration)
* Greedy decoding during evaluation
* KL prevents catastrophic drift

---

# 8. Training

Example launch:

```bash
bash scripts/train_ppo.sh
```

Typical flow:

1. VERL launches rollout workers.
2. Workers generate responses using vLLM.
3. Reward function scores outputs.
4. PPO updates actor + critic.
5. Checkpoints saved periodically.

Expected runtime:

* ~6–12 hours depending on GPU count.

---

# 9. Evaluation

Evaluation must use:

* temperature = 0
* greedy decoding
* same prompt template as training

Run:

```bash
bash scripts/evaluate.sh
```

Target metric:

```
GSM8K accuracy ≈ 80–83%
```

---

# 10. Prompt Template (Important)

Consistency matters.

Recommended:

```
System: You are a helpful assistant.
User: <GSM8K question>
Assistant:
```

Do NOT change format between training and evaluation.

---

# 11. Common Failure Modes

### Accuracy does not improve

Usually caused by:

* incorrect answer extraction
* reward always zero
* KL too large
* rollout temperature too low

### Training unstable

Reduce:

* actor_lr
* batch size

### GPU OOM

Reduce:

* max_response_length
* microbatch_size

---

# 12. Deliverables

Engineer should produce:

* trained PPO checkpoint
* evaluation script
* GSM8K accuracy report
* training logs

---

# 13. Success Criteria

✅ Base model reproduces ~69% GSM8K
✅ PPO training increases accuracy ≥10 points
✅ Final score near ~82%

---

# 14. Future Extensions (Optional)

* SA-PPO implementation
* GRPO comparison
* Multi-dataset training (GSM-Sym, MATH)
* Reward shaping (partial credit)

---

## End of Specification
