# TunedAI Labs — Causal Reasoning Model

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunedailabs/tunedailabs/blob/main/causal_reasoning_demo.ipynb)
[![Model on HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-tunedailabs%2Fcausal--reasoning--qwen--7b-blue)](https://huggingface.co/tunedailabs/causal-reasoning-qwen-7b)

A 7B open-source model fine-tuned specifically for causal reasoning. 96.96% on the CLaDDer benchmark — 25 points above GPT-4o. Run the benchmark yourself and verify.

---

## Results

| Model | CLaDDer Score | Parameters | Open Source |
|---|---|---|---|
| **TunedAI Labs — Qwen 2.5-7B (fine-tuned)** | **96.96%** | 7B | ✓ |
| GPT-4o | ~72% | unknown | ✗ |
| Claude 3.5 Sonnet | ~68% | unknown | ✗ |
| Qwen 2.5-7B (base, no fine-tuning) | ~63% | 7B | ✓ |

9,805 correct out of 10,112 questions. Full CLaDDer dataset. Results independently verified — not benchmaxxed, not overfitted.

> *"Kudos for surviving scrutiny."*
> — Independent verifier, 25 years in security and AI engineering (White House Situation Room, JP Morgan VP of cyber intelligence, Splunk Senior Solutions Architect)

---

## What is CLaDDer?

CLaDDer is a public benchmark of 10,000+ causal reasoning questions across three levels of the causal hierarchy:

- **Level 1 — Associational:** what correlates with what
- **Level 2 — Interventional:** what happens if you change something
- **Level 3 — Counterfactual:** what would have happened if things had been different

Questions use synthetic fictional variable names (yupt, jyka, kwox) with answers derived mathematically from probability parameters — making memorisation impossible. The benchmark is fully public and independently runnable.

→ [CLaDDer paper](https://arxiv.org/abs/2312.04350)

---

## Run It Yourself

**One click — no setup:**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tunedailabs/tunedailabs/blob/main/causal_reasoning_demo.ipynb)

The notebook runs on a free T4 GPU. It loads the model, runs the benchmark, and prints your score. SHA256 hash verification is included to confirm adapter integrity.

**Local setup:**

```bash
pip install transformers peft accelerate bitsandbytes
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base, "tunedailabs/causal-reasoning-qwen-7b")
```

---

## How It Was Built

Standard Qwen 2.5-7B-Instruct as the base. Fine-tuned with LoRA on synthetic causal reasoning training data — questions with machine-verified answers derived from explicit probability parameters. The training data is structured so the model must learn the underlying causal structure, not surface patterns.

The gap between 63% (base) and 96.96% (fine-tuned) is not from seeing CLaDDer questions during training. The benchmark uses fictional variable names specifically to prevent that. The improvement is from learning to reason causally.

---

## Model

HuggingFace: [tunedailabs/causal-reasoning-qwen-7b](https://huggingface.co/tunedailabs/causal-reasoning-qwen-7b)

Free to use, run anywhere, no API required.

---

## Contact

Mark Gentry — [mark.gentry@gmail.com](mailto:mark.gentry@gmail.com)  
TunedAI Labs — [tunedailabs.com](https://tunedailabs.com)
