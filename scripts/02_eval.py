#!/usr/bin/env python
"""Run one pipeline over a split, score it, and dump per-sample predictions.

Usage:
    python scripts/02_eval.py --approach ocr_first --split val [--limit 100]
    python scripts/02_eval.py --approach vlm       --split val
    python scripts/02_eval.py --approach hybrid    --split val

Predictions are written to outputs/preds/<approach>_<split>.json including the OCR
coverage bucket per sample, which 03_analysis.py consumes.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from src.config import load_config, pick_device
from src.dataset import TextVQADataset
from src.metrics import aggregate, anls, vqa_accuracy
from src.ocr import tokens_to_string
from src.ocr_quality import coverage_bucket


def load_ocr_cache(cfg, split):
    engine = cfg["ocr"]["engine"]
    path = os.path.join(cfg["paths"]["ocr_cache"], f"{split}_{engine}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"OCR cache missing: {path}. Run scripts/01_run_ocr.py first.")
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", required=True,
                    choices=["ocr_first", "vlm", "hybrid"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    device = pick_device(cfg["eval"]["device"])
    limit = args.limit if args.limit is not None else cfg["eval"]["limit"]
    ds = TextVQADataset(args.split, cfg, limit=limit)

    need_ocr = args.approach in ("ocr_first", "hybrid")
    ocr_cache = load_ocr_cache(cfg, args.split) if need_ocr else {}
    mt, mc = cfg["ocr"]["max_tokens"], cfg["ocr"]["min_confidence"]

    if args.approach == "ocr_first":
        from src.models.ocr_first import build_ocr_first
        model = build_ocr_first(cfg, device)
    elif args.approach == "vlm":
        from src.models.vlm import build_vlm
        model = build_vlm(cfg, device)
    else:
        from src.models.hybrid import build_hybrid
        model = build_hybrid(cfg, device)

    rows = []
    for i in tqdm(range(len(ds)), desc=f"eval[{args.approach}]"):
        m = ds.meta(i)
        ocr_text = ""
        if need_ocr:
            ocr_text = tokens_to_string(
                ocr_cache.get(str(m["image_id"]), []), mt, mc)

        if args.approach == "ocr_first":
            pred = model.answer(m["question"], ocr_text)
        elif args.approach == "vlm":
            pred = model.answer(ds[i]["image"], m["question"])
        else:
            pred = model.answer(ds[i]["image"], m["question"], ocr_text)

        bucket = coverage_bucket(ocr_text, m["answers"]) if m["answers"] else "unknown"
        rows.append({
            "question_id": m["question_id"],
            "question": m["question"],
            "pred": pred,
            "answers": m["answers"],
            "ocr_text": ocr_text,
            "ocr_bucket": bucket,
            "vqa_acc": vqa_accuracy(pred, m["answers"]),
            "anls": anls(pred, m["answers"]),
        })

    os.makedirs(cfg["paths"]["preds"], exist_ok=True)
    os.makedirs(cfg["paths"]["scores"], exist_ok=True)
    pred_path = os.path.join(cfg["paths"]["preds"], f"{args.approach}_{args.split}.json")
    with open(pred_path, "w") as f:
        json.dump(rows, f, indent=2)

    summary = aggregate(rows)
    summary["approach"] = args.approach
    summary["split"] = args.split
    score_path = os.path.join(cfg["paths"]["scores"], f"{args.approach}_{args.split}.json")
    with open(score_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\npredictions -> {pred_path}")


if __name__ == "__main__":
    main()
