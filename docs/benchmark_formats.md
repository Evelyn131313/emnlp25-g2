# Benchmark Generation & Evaluation Formats

This document describes the file formats and pipeline steps for each of the three benchmarks.

---

## Benchmark 1: WMT'14 DE→EN Translation

### Pipeline

```
conda activate g2
run_eval.py  →  predictions_0.jsonl ... predictions_4.jsonl
             →  all_answer.jsonl
             →  all_metrics.json

conda activate g2_eval
calculate_div.py  →  diversity metrics (printed to stdout)
```

### Step-by-step

| Step     | Script                    | What happens                                                                                                               |
| -------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| 1        | `eval/wmt/run_eval.py`  | Runs G2 for `iter_num=5` iterations. Each iteration writes one `predictions_N.jsonl` and computes BLEU + COMET inline. |
| 2 (auto) | same script               | Merges all `predictions_N.jsonl` into `all_answer.jsonl` for diversity scoring.                                        |
| 3        | `eval/calculate_div.py` | Reads `all_answer.jsonl`, computes self-BLEU, EAD, Sentence-BERT diversity. Prints to stdout, no file written.           |

### File formats

**`predictions_N.jsonl`** — one file per iteration (N = 0..4), one JSON per source sentence:

```json
{
  "source_de": "German source sentence",
  "reference_en": "Human reference translation",
  "predicted_en": "Model output translation",
  "sentence_bleu": 0.42,
  "sentence_rouge": 0.51
}
```

**`all_answer.jsonl`** — all 5 `predictions_N.jsonl` files concatenated in order (iter0 rows first, then iter1, ..., iter4). Total rows = `num_sentences × 5`. Used by `calculate_div.py`, which reassembles per-prompt response sets via `data[data_num*j+i]`.

**`all_metrics.json`** — per-iteration quality scores:

```json
{
  "0": {"AVG_BLEU": 0.31, "AVG_ROUGE1": 0.44, "AVG_COMET": 0.72},
  "1": {"AVG_BLEU": 0.28, "AVG_ROUGE1": 0.41, "AVG_COMET": 0.70},
  ...
}
```

**`calculate_div.py` stdout output** — no file written; two functions run sequentially and both print to terminal.

Metrics computed:

- **EAD** (`mean_per_input_ead_averaged_distinct_ngrams`): expectation-adjusted distinct n-grams, 0–1, higher = more diverse
- **Sentence-BERT** (`mean_per_input_sent_bert_from_sim`): semantic diversity via sentence embeddings, 0–1, higher = more diverse
- **Self-BLEU** (`qs avg scores`): `1 - self_bleu_4gram`, printed as percentage, higher = more diverse

### Output directory

```
results/wmt/theta0.3_temp1.0/
├── predictions_0.jsonl    # greedy baseline (iter 0)
├── predictions_1.jsonl    # 1st G2 generation
├── predictions_2.jsonl
├── predictions_3.jsonl
├── predictions_4.jsonl
├── all_answer.jsonl       # all iterations concatenated
└── all_metrics.json       # BLEU / ROUGE / COMET per iteration
```

---

## Benchmark 2: XLSum Multilingual Summarization

### Pipeline

Identical structure to WMT. Same script layout, same file naming.

```
conda activate g2
run_eval.py  →  predictions_0.jsonl ... predictions_4.jsonl
             →  all_answer.jsonl
             →  all_metrics.json

conda activate g2_eval
calculate_div.py  →  diversity metrics (printed to stdout)
```

### Step-by-step

| Step     | Script                     | What happens                                                                                                                  |
| -------- | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| 1        | `eval/xlsum/run_eval.py` | Runs G2 for `iter_num=5` iterations. Each iteration writes one `predictions_N.jsonl` and computes ROUGE-L + COMET inline. |
| 2 (auto) | same script                | Merges all `predictions_N.jsonl` into `all_answer.jsonl`.                                                                 |
| 3        | `eval/calculate_div.py`  | Reads `all_answer.jsonl`, computes self-BLEU, EAD, Sentence-BERT diversity. Prints to stdout, no file written.              |

### File formats

**`predictions_N.jsonl`** — same field names as WMT (note: `source_de` / `reference_en` / `predicted_en` are reused despite the task being summarization):

```json
{
  "source_de": "Full news article text",
  "reference_en": "Human-written summary",
  "predicted_en": "Model-generated summary",
  "sentence_bleu": 0.18,
  "sentence_rouge": 0.35
}
```

**`all_answer.jsonl`** — same structure as WMT: all 5 `predictions_N.jsonl` files concatenated in order.

**`all_metrics.json`**:

```json
{
  "0": {"AVG_BLEU": 0.15, "AVG_ROUGEL": 0.32, "AVG_COMET": 0.65},
  "1": {"AVG_BLEU": 0.13, "AVG_ROUGEL": 0.30, "AVG_COMET": 0.63},
  ...
}
```

**`calculate_div.py` stdout output** — identical format to WMT (same two functions, same metrics). Replace `wmt eval:` with `xlsum eval:` in the output.

### Output directory

```
results/xlsum/theta0.3_temp1.0/
├── predictions_0.jsonl    # greedy baseline (iter 0)
├── predictions_1.jsonl    # 1st G2 generation
├── predictions_2.jsonl
├── predictions_3.jsonl
├── predictions_4.jsonl
├── all_answer.jsonl
└── all_metrics.json
```

---

## Benchmark 3: NoveltyBench Open-Ended Generation

Unlike WMT/XLSum, NoveltyBench separates generation and evaluation into distinct scripts with intermediate files passed between them.

### Pipeline

```
conda activate g2
run_eval.py  →  generations.jsonl

conda activate g2_eval  (switched automatically inside run_eval.sh)
partition.py →  partitions.jsonl
score.py     →  scores.jsonl
summarize.py →  summary.json
```

### Step-by-step

| Step | Script                                  | Input                                             | Output                |
| ---- | --------------------------------------- | ------------------------------------------------- | --------------------- |
| 1    | `eval/novelty-bench/src/run_eval.py`  | HuggingFace dataset `yimingzhang/novelty-bench` | `generations.jsonl` |
| 2    | `eval/novelty-bench/src/partition.py` | `generations.jsonl`                             | `partitions.jsonl`  |
| 3    | `eval/novelty-bench/src/score.py`     | `partitions.jsonl`                              | `scores.jsonl`      |
| 4    | `eval/novelty-bench/src/summarize.py` | `scores.jsonl`                                  | `summary.json`      |

### File formats

**`generations.jsonl`** — written by `run_eval.py`, one JSON per prompt, all 10 responses in a single array:

```json
{
  "id": "...",
  "prompt": "Original question",
  "model": "meta-llama/Meta-Llama-3-8B-Instruct",
  "generations": ["response_0", "response_1", ..., "response_9"]
}
```

**`partitions.jsonl`** — written by `partition.py`, adds equivalence class labels:

```json
{
  "id": "...",
  "prompt": "...",
  "model": "...",
  "generations": ["response_0", ..., "response_9"],
  "partition": [0, 0, 1, 2, 1, 3, 2, 4, 4, 5],
  "distinct": 5
}
```

- `partition[i]`: equivalence class index for response i (two responses with the same index are semantically equivalent)
- `distinct`: `max(partition)`, i.e. number of unique classes minus 1

**`scores.jsonl`** — written by `score.py`, adds reward model scores:

```json
{
  "id": "...",
  "prompt": "...",
  "model": "...",
  "generations": ["response_0", ..., "response_9"],
  "partition": [0, 0, 1, 2, 1, 3, 2, 4, 4, 5],
  "distinct": 5,
  "scores": [7, 6, 8, 5, 7, 6, 9, 4, 5, 8],
  "mean_scores": 6.5
}
```

- `scores[i]`: quality score for response i, integer 1–10 from Skywork-Reward-Gemma-2-27B-v0.2
- `mean_scores`: mean of `scores` for this prompt

**`summary.json`** — written by `summarize.py`, final evaluation result:

```json
{
  "mean_distinct": 3.47,
  "mean_scores": 6.82
}
```

- `mean_distinct`: average `distinct` across all prompts (float; higher = more diverse)
- `mean_scores`: average `mean_scores` across all prompts (float; higher = better quality)

### Output directory

```
results/novelty/g2_theta0.3_temp1/
├── generations.jsonl    # raw model outputs (10 per prompt)
├── partitions.jsonl     # + equivalence class labels
├── scores.jsonl         # + reward model scores
└── summary.json         # final mean_distinct and mean_scores
```

---

## Key Differences Between Benchmarks

|                   | WMT                                        | XLSum                     | NoveltyBench                          |
| ----------------- | ------------------------------------------ | ------------------------- | ------------------------------------- |
| Outputs per input | 5 (one file each)                          | 5 (one file each)         | 10 (all in one file)                  |
| Quality metrics   | BLEU, COMET                                | ROUGE-L, COMET            | Reward model score (1–10)            |
| Metrics computed  | Inline in `run_eval.py`                  | Inline in `run_eval.py` | Separate `score.py`                 |
| Diversity metric  | Self-BLEU, EAD, SentBERT (via `calculate_div.py`) | Same | Self-BLEU, EAD, SentBERT (via `calculate_div.py`) + `distinct` count (via `partition.py`) |
| Final result file | `all_metrics.json` (quality) + stdout (diversity) | `all_metrics.json` (quality) + stdout (diversity) | `summary.json` (quality+distinct) + stdout (diversity) |
