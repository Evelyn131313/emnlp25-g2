# G2: Guided Generation for Enhanced Output Diversity in LLMs

## Abstract

Large Language Models (LLMs) often generate highly similar content across multiple attempts, limiting their usefulness for tasks that require diverse outputs (creative writing, reasoning, summarization, etc.). Existing solutions like temperature scaling trade diversity for quality.

**G2 (Guide-to-Generation)** is a training-free, plug-and-play method that improves output diversity while preserving quality. It uses a base generator alongside dual "Guides" — one that encourages diversity and one that suppresses it — combined through entropy-weighted decoding interventions at each token step. No fine-tuning required.

---

## How G2 Works

At each generation step, G2 runs three forward passes:

1. **Base model** — produces standard next-token logits
2. **Positive guide** — conditioned on a previous diverse output; its logits push toward different token choices
3. **Negative guide** — conditioned on the same output as a "consistency anchor"; its logits are subtracted

The final logits are:

```
logits_final = logits_base + entropy_weight × (logits_positive − logits_negative)
```

The `entropy_weight` is computed from the base model's token entropy:

- If entropy is below a threshold: weight = 0 (no intervention; model is confident)
- If entropy is above the threshold: weight = `theta` (apply the guide signal)

This means G2 only intervenes when the model is uncertain, preserving quality at confident tokens while nudging diversity at ambiguous ones.

---

## Project Structure

```
emnlp25-g2/
├── modeling/
│   └── dexperts_entropy.py        # Core G2 model (DExpertsLlama class)
│
├── eval/
│   ├── utils.py                   # Shared utilities: model loading, generation wrappers
│   ├── calculate_div.py           # Computes diversity metrics on saved outputs
│   │
│   ├── wmt/                       # WMT'14 DE→EN translation benchmark
│   │   ├── run_eval.py            # G2 evaluation script
│   │   └── run_sample.py          # Sampling-based baseline evaluation
│   │
│   ├── xlsum/                     # XLSum multilingual summarization benchmark
│   │   ├── run_eval.py            # G2 evaluation script
│   │   └── run_sample.py          # Sampling-based baseline evaluation
│   │
│   ├── novelty-bench/             # NoveltyBench open-ended generation benchmark
│   │   └── src/
│   │       ├── run_eval.py        # G2 evaluation (uses transformers)
│   │       ├── run_sample.py      # Sampling baseline (uses transformers)
│   │       ├── inference.py       # Sampling baseline (uses vLLM, faster)
│   │       ├── partition.py       # Detects semantically equivalent outputs
│   │       ├── score.py           # Scores generation novelty/quality
│   │       └── summarize.py       # Aggregates and prints final results
│   │
│   └── diversity/                 # Diversity metric implementations
│       ├── diversity_metrics.py   # DistinctNgrams, EAD, SentBERT metrics
│       ├── metric.py              # Metric base classes
│       ├── similarity_metrics.py  # NLI and embedding similarity
│       └── utils.py               # Helpers for metric computation
│
├── scripts/
│   └── eval/
│       ├── wmt/
│       │   ├── run_eval.sh        # Run G2 on WMT
│       │   ├── run_sample.sh      # Run sampling baselines on WMT
│       │   └── diversity.sh       # Compute diversity metrics on WMT outputs
│       ├── xlsum/
│       │   ├── run_eval.sh        # Run G2 on XLSum
│       │   ├── run_sample.sh      # Run sampling baselines on XLSum
│       │   └── diversity.sh       # Compute diversity metrics on XLSum outputs
│       └── novelty/
│           ├── run_eval.sh        # Run G2 on NoveltyBench + score
│           ├── run_sample.sh      # Run sampling baseline on NoveltyBench + score
│           └── diversity.sh       # Compute diversity metrics on NoveltyBench outputs
│
├── figure/
│   └── main.png                   # Paper overview figure
│
├── g2_requirements.txt            # Dependencies for inference (conda env: g2)
└── g2_eval_requirements.txt       # Dependencies for diversity evaluation (conda env: g2_eval)
```

---

## Environment Setup

> Experiments were run on 8× NVIDIA A100 80GB GPUs.

Two separate conda environments are required because `sentence-transformers` (used for diversity metrics) conflicts with the version of `transformers` used for inference.

### Inference Environment (`g2`)

Used for running G2 and all baseline generation scripts.

```bash
conda create -n g2 python=3.10
conda activate g2
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r g2_requirements.txt
```

Key packages: `transformers==4.41.0`, `vllm==0.5.0.post1`, `accelerate`, `flash-attn`, `datasets`

### Evaluation Environment (`g2_eval`)

Used only for diversity metric computation (self-BLEU, EAD, Sentence-BERT).

```bash
conda create -n g2_eval python=3.10
conda activate g2_eval
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r g2_eval_requirements.txt
```

Key packages: `sentence-transformers==4.0.2`, `transformers==4.45.1`

---

## Running the Experiments

All scripts must be run from the **project root** (`emnlp25-g2/`). The default model is `meta-llama/Meta-Llama-3-8B-Instruct`; edit the scripts to change it.

### Key Parameters

| Parameter                       | Description                                     | Default                                  |
| ------------------------------- | ----------------------------------------------- | ---------------------------------------- |
| `--theta`                     | Guide signal strength (entropy weight cap)      | `0.3`                                  |
| `--temperature`               | Sampling temperature                            | `1.0`                                  |
| `--iter_num`                  | Number of diverse outputs to generate per input | `5` (WMT/XLSum), `10` (NoveltyBench) |
| `--eval_batch_size`           | Batch size for inference                        | `10` (WMT), `5` (XLSum)              |
| `--save_dir` / `--eval-dir` | Output directory for results                    | varies                                   |
| `--model_name_or_path`        | HuggingFace model ID or local path              | `meta-llama/Meta-Llama-3-8B-Instruct`  |

---

### Benchmark 1: WMT'14 DE→EN Translation

**Dataset**: Loaded automatically from HuggingFace (`wmt14`, `de-en`, test split).
**Task**: Translate German sentences to English; generate 5 diverse translations per sentence.
**Quality metrics**: BLEU, COMET
**Diversity metrics**: Self-BLEU, EAD, Sentence-BERT similarity

#### Step 1 — Run G2

```bash
conda activate g2
bash scripts/eval/wmt/run_eval.sh
```

Equivalent command:

```bash
export CUDA_VISIBLE_DEVICES=0
python -m eval.wmt.run_eval \
    --save_dir results/wmt/theta0.3_temp1.0 \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --eval_batch_size 10 \
    --temperature 1.0 \
    --theta 0.3
```

#### Step 2 — Run Sampling Baselines (optional)

```bash
conda activate g2
bash scripts/eval/wmt/run_sample.sh
```

This evaluates temperature sampling, top-k, top-p, and min-p baselines.

#### Step 3 — Compute Diversity Metrics

```bash
conda activate g2_eval
bash scripts/eval/wmt/diversity.sh
```

Equivalent command:

```bash
python eval/calculate_div.py \
    --file results/wmt/theta0.3_temp1.0/all_answer.jsonl \
    --task wmt
```

#### Output Files

```
results/wmt/theta0.3_temp1.0/
├── output_iter0.jsonl       # Baseline (greedy) translations
├── output_iter1.jsonl       # 1st diverse generation
├── output_iter2.jsonl       # 2nd diverse generation
├── output_iter3.jsonl       # 3rd diverse generation
├── output_iter4.jsonl       # 4th diverse generation
├── all_answer.jsonl         # All outputs merged (used for diversity scoring)
└── all_metrics.json         # BLEU and COMET scores per iteration
```

---

### Benchmark 2: XLSum Multilingual Summarization

**Dataset**: Loaded automatically from HuggingFace (`GEM/xlsum`).
**Task**: Summarize news articles; generate 5 diverse summaries per article.
**Quality metrics**: ROUGE-L, BERTScore
**Diversity metrics**: Self-BLEU, EAD, Sentence-BERT similarity

#### Step 1 — Run G2

```bash
conda activate g2
bash scripts/eval/xlsum/run_eval.sh
```

Equivalent command:

```bash
export CUDA_VISIBLE_DEVICES=0
python -m eval.xlsum.run_eval \
    --save_dir results/xlsum/theta0.3_temp1.0 \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --eval_batch_size 5 \
    --temperature 1.0 \
    --theta 0.3
```

#### Step 2 — Run Sampling Baselines (optional)

```bash
conda activate g2
bash scripts/eval/xlsum/run_sample.sh
```

#### Step 3 — Compute Diversity Metrics

```bash
conda activate g2_eval
bash scripts/eval/xlsum/diversity.sh
```

Equivalent command:

```bash
python eval/calculate_div.py \
    --file results/xlsum/theta0.3_temp1.0 \
    --task xlsum
```

#### Output Files

```
results/xlsum/theta0.3_temp1.0/
├── output_iter0.jsonl       # Baseline summaries
├── output_iter1.jsonl       # 1st diverse generation
├── ...
└── all_metrics.json         # ROUGE-L and BERTScore per iteration
```

---

### Benchmark 3: NoveltyBench Open-Ended Generation

**Dataset**: `curated` split of NoveltyBench (`yimingzhang/novelty-bench` on HuggingFace), spanning creative writing, randomness, factual knowledge, and subjective opinion generation.
**Task**: Generate 10 diverse outputs per prompt; evaluated on how many are semantically non-equivalent.
**Metrics**: NoveltyBench score (fraction of semantically distinct outputs)

The NoveltyBench pipeline has two stages: **generation** (Step 1) and **evaluation** (Steps 2–4).

#### Step 1 — Run G2 Generation + Full Pipeline

```bash
conda activate g2
bash scripts/eval/novelty/run_eval.sh
```

This script runs generation then automatically switches to `g2_eval` for scoring:

```bash
# Generation (g2 env)
export CUDA_VISIBLE_DEVICES=0
python eval/novelty-bench/src/run_eval.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --data curated \
    --eval-dir results/novelty/g2_theta0.3_temp1 \
    --iter_num 10 \
    --temperature 1 \
    --theta 0.3

# Evaluation (g2_eval env)
python eval/novelty-bench/src/partition.py \
    --eval-dir results/novelty/g2_theta0.3_temp1 \
    --alg classifier

python eval/novelty-bench/src/score.py \
    --eval-dir results/novelty/g2_theta0.3_temp1 \
    --patience 0.8

python eval/novelty-bench/src/summarize.py \
    --eval-dir results/novelty/g2_theta0.3_temp1
```

#### Step 2 — Run Sampling Baselines (optional)

```bash
conda activate g2
bash scripts/eval/novelty/run_sample.sh
```

> **Note**: This uses `inference.py` (vLLM backend) by default, which is faster. To use the `transformers` backend instead, edit the script to call `run_sample.py`.

#### Step 3 — Compute Diversity Metrics

```bash
conda activate g2_eval
bash scripts/eval/novelty/diversity.sh
```

Equivalent command:

```bash
python eval/calculate_div.py \
    --file results/novelty/g2_theta0.3_temp1.0 \
    --task curated
```

#### Evaluation Pipeline Details

| Script           | What it does                                                                    |
| ---------------- | ------------------------------------------------------------------------------- |
| `partition.py` | Groups generated outputs by semantic equivalence using an NLI classifier        |
| `score.py`     | Computes the NoveltyBench score: how many distinct correct solutions were found |
| `summarize.py` | Aggregates results across all problems and prints a summary table               |

---

## Reproducing All Results

Quick reference for reproducing the full set of paper results:

```bash
# --- Generation (g2 env) ---
conda activate g2

bash scripts/eval/wmt/run_eval.sh        # WMT G2
bash scripts/eval/wmt/run_sample.sh      # WMT baselines

bash scripts/eval/xlsum/run_eval.sh      # XLSum G2
bash scripts/eval/xlsum/run_sample.sh    # XLSum baselines

bash scripts/eval/novelty/run_eval.sh    # NoveltyBench G2 + scoring
bash scripts/eval/novelty/run_sample.sh  # NoveltyBench baselines + scoring

# --- Diversity Metrics (g2_eval env) ---
conda activate g2_eval

bash scripts/eval/wmt/diversity.sh
bash scripts/eval/xlsum/diversity.sh
bash scripts/eval/novelty/diversity.sh
```

---

## Acknowledgments

We thank the following open-source projects:

- [Proxy-tuning](https://github.com/alisawuffles/proxy-tuning) — for the DExperts-style decoding framework that inspired G2's architecture
- [NoveltyBench](https://github.com/novelty-bench/novelty-bench) — for the reasoning diversity benchmark and evaluation pipeline
- [RLHF Gen Diversity](https://github.com/facebookresearch/rlhf-gen-div) — for the diversity metric implementations (self-BLEU, EAD, Sentence-BERT)

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{g2-guided-generation-2025,
    title     = {G2: Guided Generation for Enhanced Output Diversity in LLMs},
    author    = {Zhiwen Ruan and Yixia Li and Yefeng Liu and Yun Chen and
                 Weihua Luo and Peng Li and Yang Liu and Guanhua Chen},
    booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
    year      = {2025},
    publisher = {Association for Computational Linguistics}
}
```
