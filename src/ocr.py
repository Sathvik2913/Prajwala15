"""OCR wrappers. Returns a list of {text, conf, bbox} tokens per image.

Two engines:
  - easyocr  : deep OCR, better on scene text, needs torch (GPU helps)
  - tesseract: fast CPU baseline, noisier on scene text -> good for the
               "noisy OCR" comparison the project asks for.
"""
from PIL import Image


class OCREngine:
    def read(self, image: Image.Image) -> list[dict]:
        raise NotImplementedError


class EasyOCR(OCREngine):
    def __init__(self, langs=("en",), gpu=False):
        import easyocr
        self.reader = easyocr.Reader(list(langs), gpu=gpu)

    def read(self, image: Image.Image) -> list[dict]:
        import numpy as np
        res = self.reader.readtext(np.array(image))
        out = []
        for bbox, text, conf in res:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            out.append({
                "text": text,
                "conf": float(conf),
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
            })
        return out


class Tesseract(OCREngine):
    def __init__(self, langs=("en",)):
        self.lang = "+".join(langs)

    def read(self, image: Image.Image) -> list[dict]:
        import pytesseract
        from pytesseract import Output
        d = pytesseract.image_to_data(image, lang=self.lang, output_type=Output.DICT)
        out = []
        for i, text in enumerate(d["text"]):
            text = text.strip()
            if not text:
                continue
            conf = float(d["conf"][i]) / 100.0 if d["conf"][i] not in ("-1", -1) else 0.0
            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            out.append({"text": text, "conf": conf, "bbox": [x, y, x + w, y + h]})
        return out


def build_engine(name: str, langs=("en",), gpu=False) -> OCREngine:
    name = name.lower()
    if name == "easyocr":
        return EasyOCR(langs, gpu=gpu)
    if name == "tesseract":
        return Tesseract(langs)
    raise ValueError(f"unknown OCR engine: {name}")


def tokens_to_string(tokens: list[dict], max_tokens: int, min_conf: float) -> str:
    kept = [t["text"] for t in tokens if t["conf"] >= min_conf][:max_tokens]
    return " ".join(kept)
