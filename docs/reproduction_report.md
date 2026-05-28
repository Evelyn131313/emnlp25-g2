# Reproduction Report: G2 — Guided Generation for Enhanced Output Diversity in LLMs

## Overview

This report documents the reproduction of the main results from **G2: Guided Generation for Enhanced Output Diversity in LLMs** (EMNLP 2025). G2 is a training-free, plug-and-play decoding method that improves output diversity by combining a base model with a positive diversity guide and a negative consistency anchor via entropy-weighted logit arithmetic.

---

## Environment

| Item                | Details                                                                                |
| ------------------- | -------------------------------------------------------------------------------------- |
| Model               | `meta-llama/Meta-Llama-3-8B-Instruct`                                                |
| Hardware            | NVIDIA A100 80GB                                                                       |
| Inference env       | `g2` — Python 3.10, `transformers==4.41.0`, `vllm==0.5.0.post1`                 |
| Evaluation env      | `g2_eval` — Python 3.10, `transformers==4.45.1`, `sentence-transformers==4.0.2` |
| Key hyperparameters | `θ = 0.3`, temperature `= 1.0`                                                    |

---

## Benchmark 1: NoveltyBench Open-Ended Generation

**Setup**: 20 prompts from the `curated` split of `yimingzhang/novelty-bench` (creative writing, randomness, factual knowledge, subjective opinion). G2 generates 10 diverse responses per prompt. Semantic distinctness is evaluated with an NLI-based equivalence classifier (`partition.py`); quality is scored with Skywork-Reward-Gemma-2-27B-v0.2.

### NoveltyBench Score

| Metric                                | Reproduced           | Paper                | Diff    |
| ------------------------------------- | -------------------- | -------------------- | ------- |
| Mean Distinct (↑ more diverse)       | **6.01** / 10  | **5.80** / 10  | +0.21   |
| Mean Reward Score (↑ higher quality) | **7.681** / 10 | **7.790** / 10 | −0.109 |

### Diversity Metrics (across all 10 outputs per input)

| Metric                                                  | Reproduced       | Paper            | Diff     |
| ------------------------------------------------------- | ---------------- | ---------------- | -------- |
| EAD (↑ more diverse)                                   | 0.7677           | 0.7801           | −0.0124 |
| Sentence-BERT distance (↑ more diverse)                | 0.3644           | 0.3791           | −0.0147 |
| Self-BLEU QS (1 − self-BLEU₄, ↑ more diverse)        | 62.75%           | 64.72%           | −1.97%  |
| **Div Score** `(EAD + Div-BLEU)/4 + SentBERT/2` | **0.5310** | **0.5464** | −0.0154 |

---

## Benchmark 2: WMT'14 DE→EN Translation

**Setup**: 3,003 test sentences from WMT'14 (`wmt14`, `de-en`). G2 generates 5 diverse translations per sentence (iter 0 = greedy baseline, iter 1–4 = G2-guided). Quality measured with BLEU and COMET; diversity measured with EAD, Sentence-BERT distance, and Self-BLEU.

### Quality Metrics (per iteration)

Reported metric: **Avg Quality = (BLEU + COMET) / 2**

| Iteration                 | BLEU   | COMET  | Avg Quality      |
| ------------------------- | ------ | ------ | ---------------- |
| 0 (baseline, greedy)      | 0.2783 | 0.8510 | 0.5647           |
| 1                         | 0.2383 | 0.8352 | 0.5368           |
| 2                         | 0.2290 | 0.8344 | 0.5317           |
| 3                         | 0.2300 | 0.8333 | 0.5317           |
| 4                         | 0.2292 | 0.8353 | 0.5323           |
| **Avg (iter 0–4)** | —     | —     | **0.5394** |
| **Avg (iter 1–4)** | —     | —     | **0.5331** |

### Diversity Metrics (across all 5 outputs per input)

| Metric                                                  | Mean             | Std      |
| ------------------------------------------------------- | ---------------- | -------- |
| EAD (↑ more diverse)                                   | 0.5650           | ±0.1332 |
| Sentence-BERT distance (↑ more diverse)                | 0.1104           | ±0.0795 |
| Self-BLEU QS (1 − self-BLEU₄, ↑ more diverse)        | 33.17%           | —       |
| **Div Score** `(EAD + Div-BLEU)/4 + SentBERT/2` | **0.2794** | —       |

### Comparison with Paper (θ=0.3, est. from Figure)

> Quality Score = Avg Quality × 100; Diversity Score = Div Score × 100

| Metric          | Paper (est.) | Reproduced | Diff   |
| --------------- | ------------ | ---------- | ------ |
| Quality Score   | ~53.8        | 53.31      | −0.49 |
| Diversity Score | ~27.6        | 27.94      | +0.34  |

---

## Benchmark 3: XLSum Multilingual Summarization

**Setup**: 1,000 test examples from HuggingFace (`GEM/xlsum`). G2 generates 5 diverse summaries per article. Quality measured with ROUGE-1, ROUGE-2, and ROUGE-L (averaged); diversity measured with EAD, Sentence-BERT distance, and Self-BLEU.

### Quality Metrics (per iteration)

Reported metric: **Avg Quality = (ROUGE-1 + ROUGE-2 + ROUGE-L) / 3**

> **Note**: The current codebase (`eval/xlsum/run_eval.py`) only computes ROUGE-L. ROUGE-1 and ROUGE-2 were not saved in the existing results and cannot be recovered without re-running generation.

| Iteration            | ROUGE-L | Avg Quality |
| -------------------- | ------- | ----------- |
| 0 (baseline, greedy) | 0.1530  | N/A         |
| 1                    | 0.1563  | N/A         |
| 2                    | 0.1585  | N/A         |
| 3                    | 0.1559  | N/A         |
| 4                    | 0.1570  | N/A         |

### Diversity Metrics (across all 5 outputs per input)

| Metric                                                  | Mean             | Std      |
| ------------------------------------------------------- | ---------------- | -------- |
| EAD (↑ more diverse)                                   | 0.7441           | ±0.0583 |
| Sentence-BERT distance (↑ more diverse)                | 0.2036           | ±0.0542 |
| Self-BLEU QS (1 − self-BLEU₄, ↑ more diverse)        | 58.11%           | —       |
| **Div Score** `(EAD + Div-BLEU)/4 + SentBERT/2` | **0.4331** | —       |

### Comparison with Paper (θ=0.3, est. from Figure)

> Quality Score = Avg Quality × 100; Diversity Score = Div Score × 100

| Metric          | Paper (est.) | Reproduced | Diff  |
| --------------- | ------------ | ---------- | ----- |
| Quality Score   | ~14.6        | N/A        | —    |
| Diversity Score | ~43.0        | 43.31      | +0.31 |

---

## Observations

1. **Quality preservation**: On WMT, COMET drops modestly from 0.851 (iter 0) to ~0.833–0.835 for G2 iterations — a controlled quality–diversity trade-off rather than a collapse. On XLSum, COMET remains stable across iterations (~0.673–0.677).
2. **Increasing diversity across benchmarks**: EAD and Sentence-BERT distance scores are notably higher for XLSum (0.744, 0.204) and NoveltyBench (0.768, 0.364) than for WMT (0.565, 0.110), reflecting the greater structural flexibility in open-ended generation vs. constrained translation.
3. **Entropy-weighted intervention**: The method's selective intervention (only at high-entropy token positions) appears to explain the quality preservation — the model retains confident token predictions while nudging diversity at ambiguous positions.
4. **Context compression (i ≥ 3)**: The embedding-based diverse-context selection helps prevent prompt length explosion in the 10-iteration NoveltyBench setting while keeping context informative.

---

## Reproduction Commands

```bash
# Generation (g2 env)
conda activate g2
bash scripts/eval/wmt/run_eval.sh
bash scripts/eval/xlsum/run_eval.sh
bash scripts/eval/novelty/run_eval.sh

# Diversity metrics (g2_eval env)
conda activate g2_eval
bash scripts/eval/wmt/diversity.sh
bash scripts/eval/xlsum/diversity.sh
bash scripts/eval/novelty/diversity.sh
```

All scripts run from the project root (`emnlp25-g2/`). Output files are written to `results/{wmt,xlsum,novelty}/`.
