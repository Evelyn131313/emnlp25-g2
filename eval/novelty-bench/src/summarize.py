# Pipeline: run_eval.py → partition.py → score.py → summarize.py
#
# Stage 3 of 3 (aggregation): collapses per-instance results into a single
# summary (mean_distinct = avg diversity, mean_scores = avg quality).
#
# Reads:  {eval_dir}/scores.jsonl
#   {
#     "id": "...",
#     "prompt": "...",
#     "model": "...",
#     "generations": ["response1", "response2", ...],
#     "partition": [0, 0, 1, 2, 1],
#     "distinct": 2,
#     "scores": [7, 6, 8, 5, 7],
#     "mean_scores": 6.6
#   }
#
# Writes: {eval_dir}/summary.json  (final evaluation result)
#   {
#     "mean_distinct": 3.47,   # avg distinct across all prompts (float)
#     "mean_scores": 6.82      # avg quality score across all prompts (float)
#   }

import argparse
import json
import os

import numpy as np
import pandas as pd


def summarize(df: pd.DataFrame, decay_rate=0.5) -> dict:
    summary = {}

    summary["mean_distinct"] = np.mean(df["distinct"])
    summary["mean_scores"] = np.mean(df["mean_scores"])

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir", help="Directory containing evaluation files", required=True
    )
    args = parser.parse_args()

    eval_dir = args.eval_dir
    df = pd.read_json(os.path.join(eval_dir, "scores.jsonl"), lines=True)
    summary = summarize(df)
    with open(os.path.join(eval_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
