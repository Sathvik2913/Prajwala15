#!/usr/bin/env python
"""Run OCR over a split once and cache the result.

Usage:
    python scripts/01_run_ocr.py --split val --engine easyocr [--gpu]
    python scripts/01_run_ocr.py --split val --engine easyocr --preprocess
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from src.config import load_config, ocr_cache_path
from src.dataset import TextVQADataset
from src.ocr import build_engine, prepare_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--engine", default=None, help="easyocr|tesseract|paddleocr")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--preprocess", action="store_true",
                    help="apply denoise/contrast/deskew before OCR")
    ap.add_argument("--force", action="store_true",
                    help="re-run OCR even if image is already cached")
    args = ap.parse_args()

    cfg = load_config()
    if args.preprocess:
        cfg["ocr"]["preprocess"] = True
    engine_name = args.engine or cfg["ocr"]["engine"]
    ds = TextVQADataset(args.split, cfg, limit=args.limit)
    engine = build_engine(engine_name, tuple(cfg["ocr"]["langs"]), gpu=args.gpu)

    out_path = ocr_cache_path(cfg, args.split, engine_name, cfg["ocr"].get("preprocess"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cache = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            cache = json.load(f)

    for i in tqdm(range(len(ds)), desc=f"OCR[{engine_name}]"):
        m = ds.meta(i)
        key = str(m["image_id"])
        if key in cache and not args.force:
            continue
        try:
            img = prepare_image(ds[i]["image"], cfg)
            cache[key] = engine.read(img)
        except FileNotFoundError:
            cache[key] = []
        except Exception as e:
            print(f"warn: OCR failed for {key}: {e}")
            cache[key] = []
        if i % 200 == 0:
            with open(out_path, "w") as f:
                json.dump(cache, f)

    with open(out_path, "w") as f:
        json.dump(cache, f)
    print(f"wrote {out_path}  ({len(cache)} images)")


if __name__ == "__main__":
    main()
