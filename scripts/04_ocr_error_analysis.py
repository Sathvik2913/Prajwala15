#!/usr/bin/env python
"""Member 1 deliverable: low-confidence OCR error analysis.

Reads OCR cache + optional ocr_first predictions and produces:
  - outputs/member1/low_confidence_tokens.csv
  - outputs/member1/low_confidence_summary.json
  - outputs/member1/low_confidence_examples.png  (sample images with stats)

Usage:
    python scripts/04_ocr_error_analysis.py --split val
    python scripts/04_ocr_error_analysis.py --split val --conf-threshold 0.5
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.config import load_config, ocr_cache_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf-threshold", type=float, default=0.5)
    ap.add_argument("--engine", default=None)
    ap.add_argument("--preprocess", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.preprocess:
        cfg["ocr"]["preprocess"] = True
    engine = args.engine or cfg["ocr"]["engine"]

    cache_path = ocr_cache_path(cfg, args.split, engine,
                                cfg["ocr"].get("preprocess"))
    if not os.path.exists(cache_path):
        print(f"Missing OCR cache: {cache_path}")
        return

    with open(cache_path) as f:
        ocr_data = json.load(f)

    low_conf = []
    stats = {"images": len(ocr_data), "tokens": 0, "low_conf_tokens": 0,
             "images_with_text": 0, "images_empty": 0}

    for img_id, tokens in ocr_data.items():
        if tokens:
            stats["images_with_text"] += 1
        else:
            stats["images_empty"] += 1
        for t in tokens:
            stats["tokens"] += 1
            if t["conf"] < args.conf_threshold:
                stats["low_conf_tokens"] += 1
                low_conf.append({
                    "image_id": img_id,
                    "text": t["text"],
                    "conf": t["conf"],
                    "bbox": t.get("bbox", []),
                })

    out_dir = os.path.join("outputs", "member1")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(low_conf)
    csv_path = os.path.join(out_dir, "low_confidence_tokens.csv")
    df.to_csv(csv_path, index=False)

    pred_path = os.path.join(cfg["paths"]["preds"], f"ocr_first_{args.split}.json")
    pred_summary = {}
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            preds = json.load(f)
        pred_df = pd.DataFrame(preds)
        pred_summary = {
            "n": len(pred_df),
            "vqa_accuracy": float(pred_df["vqa_acc"].mean()),
            "f1": float(pred_df["f1"].mean()) if "f1" in pred_df else None,
            "exact_match": float(pred_df["exact_match"].mean())
            if "exact_match" in pred_df else None,
        }

    summary = {
        "split": args.split,
        "ocr_cache": cache_path,
        "conf_threshold": args.conf_threshold,
        "ocr_stats": stats,
        "low_conf_fraction": (stats["low_conf_tokens"] / stats["tokens"]
                              if stats["tokens"] else 0),
        "baseline": pred_summary,
    }
    summary_path = os.path.join(out_dir, "low_confidence_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Member 1 OCR error analysis ===")
    print(f"OCR cache: {cache_path}")
    print(f"Images: {stats['images']}  |  with text: {stats['images_with_text']}  "
          f"|  empty: {stats['images_empty']}")
    print(f"Tokens: {stats['tokens']}  |  low-confidence (<{args.conf_threshold}): "
          f"{stats['low_conf_tokens']}")
    if pred_summary:
        print(f"\nOCR-first baseline (n={pred_summary['n']}):")
        print(f"  VQA accuracy: {pred_summary['vqa_accuracy']:.3f}")
        if pred_summary.get("f1") is not None:
            print(f"  F1 (exact match): {pred_summary['f1']:.3f}")
    print(f"\nsaved {csv_path}")
    print(f"saved {summary_path}")

    if len(df):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].hist(df["conf"], bins=20, color="coral", edgecolor="black")
            axes[0].set_xlabel("confidence")
            axes[0].set_ylabel("count")
            axes[0].set_title(f"Low-confidence tokens (<{args.conf_threshold})")

            top = df["text"].str.len().clip(upper=30)
            axes[1].hist(top, bins=15, color="steelblue", edgecolor="black")
            axes[1].set_xlabel("token length (capped)")
            axes[1].set_title("Token length distribution")
            fig.tight_layout()
            png = os.path.join(out_dir, "low_confidence_examples.png")
            fig.savefig(png, dpi=150)
            print(f"saved {png}")
        except Exception as e:
            print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
