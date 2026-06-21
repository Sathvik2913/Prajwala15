#!/usr/bin/env python
"""Compare approaches and break results down by OCR quality.

Reads outputs/preds/<approach>_<split>.json for each available approach and produces:
  - outputs/comparison_<split>.csv   (overall + per-bucket accuracy)
  - outputs/comparison_<split>.png   (grouped bar chart)
  - a printed summary table

Usage:
    python scripts/03_analysis.py --split val
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.config import load_config

BUCKETS = ["full", "partial", "none"]
APPROACHES = ["ocr_first", "vlm", "hybrid"]


def load_rows(cfg, approach, split):
    p = os.path.join(cfg["paths"]["preds"], f"{approach}_{split}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    cfg = load_config()

    table = {}            # approach -> {overall, full, partial, none}
    counts = {}
    for ap_name in APPROACHES:
        rows = load_rows(cfg, ap_name, args.split)
        if rows is None:
            continue
        df = pd.DataFrame(rows)
        rec = {"overall": df["vqa_acc"].mean()}
        for b in BUCKETS:
            sub = df[df["ocr_bucket"] == b]
            rec[b] = sub["vqa_acc"].mean() if len(sub) else float("nan")
            counts[b] = len(sub)
        table[ap_name] = rec

    if not table:
        print("No prediction files found. Run scripts/02_eval.py first.")
        return

    out = pd.DataFrame(table).T[["overall"] + BUCKETS]
    out.index.name = "approach"

    os.makedirs("outputs", exist_ok=True)
    csv_path = f"outputs/comparison_{args.split}.csv"
    out.to_csv(csv_path)

    print("\n=== TextVQA accuracy by approach and OCR coverage of GT answer ===")
    print(f"(bucket sizes: {counts})\n")
    print(out.round(3).to_string())
    print(f"\nsaved {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        cols = ["overall"] + BUCKETS
        x = np.arange(len(cols))
        w = 0.8 / max(len(out), 1)
        fig, axu = plt.subplots(figsize=(9, 5))
        for i, (name, row) in enumerate(out.iterrows()):
            axu.bar(x + i * w, [row[c] for c in cols], w, label=name)
        axu.set_xticks(x + w * (len(out) - 1) / 2)
        axu.set_xticklabels(cols)
        axu.set_ylabel("VQA accuracy")
        axu.set_title(f"TextVQA ({args.split}): approach × OCR coverage")
        axu.legend()
        fig.tight_layout()
        png = f"outputs/comparison_{args.split}.png"
        fig.savefig(png, dpi=150)
        print(f"saved {png}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
