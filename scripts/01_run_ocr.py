#!/usr/bin/env python
"""Run OCR over a split once and cache the result, so eval is fast and repeatable.

Usage:
    python scripts/01_run_ocr.py --split val --engine easyocr [--limit 50] [--gpu]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from src.config import load_config
from src.dataset import TextVQADataset
from src.ocr import build_engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--engine", default=None, help="easyocr|tesseract (default: config)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    engine_name = args.engine or cfg["ocr"]["engine"]
    ds = TextVQADataset(args.split, cfg, limit=args.limit)
    engine = build_engine(engine_name, tuple(cfg["ocr"]["langs"]), gpu=args.gpu)

    out_dir = cfg["paths"]["ocr_cache"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.split}_{engine_name}.json")

    cache = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            cache = json.load(f)

    for i in tqdm(range(len(ds)), desc=f"OCR[{engine_name}]"):
        m = ds.meta(i)
        key = str(m["image_id"])
        if key in cache:
            continue
        try:
            img = ds[i]["image"]
            cache[key] = engine.read(img)
        except FileNotFoundError:
            cache[key] = []
        if i % 200 == 0:
            with open(out_path, "w") as f:
                json.dump(cache, f)

    with open(out_path, "w") as f:
        json.dump(cache, f)
    print(f"wrote {out_path}  ({len(cache)} images)")


if __name__ == "__main__":
    main()
