# TunedAI — Causal Reasoning Benchmark

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mgentry11/tunedai/blob/main/causal_reasoning_demo.ipynb)

We fine-tuned Qwen 2.5-7B on causal reasoning and scored **96.96%** on the CLadder benchmark — within 1.6% of the theoretical ceiling. GPT-4o scores ~72% on the same test. Base Qwen scores ~62%.

**This repo lets you verify that claim yourself.**

---

## The Benchmark

CLadder tests causal reasoning across all three levels of Judea Pearl's causal hierarchy:

| Model | CLadder Score | Type |
|---|---|---|
| **TunedAI Causal Model** (Qwen 2.5-7B) | **96.96%** | Fine-tuned |
| GPT-4o | ~72% | General purpose |
| Claude 3.5 Sonnet | ~68% | General purpose |
| Base Qwen 2.5-7B | ~62% | Untuned |

CLadder is a public benchmark — [verify it independently](https://github.com/causalNLP/cladder).

---

## Run It Yourself

Click the Colab badge above. The notebook:

1. Loads base Qwen 2.5-7B and the TunedAI fine-tuned adapter
2. Optionally loads GPT-4o and Claude 3.5 (bring your own API keys)
3. Runs all models on the same questions and prints results side-by-side
4. Lets you type your own causal reasoning questions

**No API keys required** to compare base Qwen vs TunedAI. Keys are only needed if you want the GPT-4 and Claude columns.

Runtime: T4 GPU (free tier), ~2 min to load.

---

## The Test Questions

All questions come from **pre-AI sources** — classic texts published decades before large language models existed. The correct answers were established by human experts. This rules out pattern-matching from internet training data.

| Test | Source | Year | Rung |
|---|---|---|---|
| Simpson's Paradox — kidney stone treatment | E.H. Simpson | 1951 | 2 — Intervention |
| Cholera and the Broad Street pump | John Snow | 1854 | 1 → 2 |
| Handwashing and childbed fever | Ignaz Semmelweis | 1847 | 1 → 2 |
| Smoking and lung cancer — no RCT | Bradford Hill | 1965 | 1 — Causal criteria |
| Billiard balls and overdetermination | David Hume | 1748 | 3 — Counterfactual |
| Why randomization enables causal claims | R.A. Fisher | 1935 | 2 — Randomization |

---

## Share Your Results

Run the notebook and post what you got in [Issues](https://github.com/mgentry11/tunedai/issues).

Tell us:
- Which models you ran (base Qwen, GPT-4, Claude, all four)
- Which question you found most interesting
- Where TunedAI got it right and the others didn't — or where it surprised you

We'll compile results from independent runs and post a summary.

---

## About TunedAI

We fine-tune open-source LLMs for real-world reasoning tasks. Causal reasoning is our flagship specialty — but we tune for any domain where getting the reasoning right actually matters.

**Want this for your domain?** → [tunedai.ai](https://tunedai.ai)
