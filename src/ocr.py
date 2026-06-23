"""OCR wrappers. Returns a list of {text, conf, bbox} tokens per image.

Engines:
  - easyocr   : deep OCR, better on scene text (GPU helps)
  - tesseract : fast CPU baseline
  - paddleocr : PaddleOCR (good on scene text, angle classification)
"""
from PIL import Image
import shutil

from .preprocess import preprocess_image


class OCREngine:
    def read(self, image: Image.Image) -> list[dict]:
        raise NotImplementedError


class EasyOCR(OCREngine):
    def __init__(self, langs=("en",), gpu=False, model_dir=None):
        import os
        import easyocr
        if model_dir is None:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.environ.get("EASYOCR_MODEL_DIR",
                                       os.path.join(root, ".easyocr_models"))
        os.makedirs(model_dir, exist_ok=True)
        self.reader = easyocr.Reader(
            list(langs), gpu=gpu, model_storage_directory=model_dir, verbose=False
        )

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
    def __init__(self, langs=("eng",)):
        import os
        import pytesseract
        self.lang = "+".join(langs)
        if not shutil.which("tesseract"):
            for candidate in (
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            ):
                if os.path.exists(candidate):
                    pytesseract.pytesseract.tesseract_cmd = candidate
                    break
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tessdata_dir = os.path.join(root, "tessdata")
        if os.path.isdir(tessdata_dir):
            os.environ["TESSDATA_PREFIX"] = tessdata_dir + os.sep

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


class PaddleOCRReader(OCREngine):
    def __init__(self, langs=("en",), gpu=False):
        from paddleocr import PaddleOCR
        lang = "en" if langs in (("en",), ("eng",)) else langs[0]
        self.reader = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=gpu,
            show_log=False,
        )

    def read(self, image: Image.Image) -> list[dict]:
        import numpy as np
        res = self.reader.ocr(np.array(image.convert("RGB")), cls=True)
        out = []
        if not res or res[0] is None:
            return out
        for line in res[0]:
            bbox, (text, conf) = line
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            out.append({
                "text": text,
                "conf": float(conf),
                "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
            })
        return out


def _normalize_langs(name: str, langs) -> tuple[str, ...]:
    """Map config langs to each engine's expected codes (config uses ISO-style 'en')."""
    name = name.lower()
    out = []
    for lang in langs:
        code = str(lang).lower()
        if name == "tesseract":
            out.append("eng" if code in ("en", "eng") else code)
        elif name in ("easyocr", "paddleocr"):
            out.append("en" if code in ("en", "eng") else code)
        else:
            out.append(code)
    return tuple(out) or ("en",)


def build_engine(name: str, langs=("eng",), gpu=False) -> OCREngine:
    name = name.lower()
    langs = _normalize_langs(name, langs)
    if name == "easyocr":
        return EasyOCR(langs, gpu=gpu)
    if name == "tesseract":
        return Tesseract(langs)
    if name == "paddleocr":
        return PaddleOCRReader(langs, gpu=gpu)
    raise ValueError(f"unknown OCR engine: {name}")


def prepare_image(image: Image.Image, cfg: dict) -> Image.Image:
    """Optionally preprocess before OCR (controlled by cfg['ocr']['preprocess'])."""
    o = cfg.get("ocr", {})
    if not o.get("preprocess", False):
        return image
    pp = o.get("preprocess_opts", {})
    return preprocess_image(
        image,
        denoise=pp.get("denoise", True),
        enhance=pp.get("enhance", True),
        deskew=pp.get("deskew", True),
    )


def tokens_to_string(tokens: list[dict], max_tokens: int, min_conf: float) -> str:
    kept = [t["text"] for t in tokens if t["conf"] >= min_conf][:max_tokens]
    return " ".join(kept)
