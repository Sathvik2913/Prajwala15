import os
import yaml

_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "default.yaml")


def load_config(path: str = _DEFAULT) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def ocr_cache_path(cfg: dict, split: str, engine: str | None = None,
                   preprocess: bool | None = None) -> str:
    """Path for cached OCR JSON (optionally tagged when preprocessing is on)."""
    ocr = cfg["ocr"]
    engine = engine or ocr["engine"]
    if preprocess is None:
        preprocess = ocr.get("preprocess", False)
    tag = f"{split}_{engine}"
    if preprocess:
        tag += "_preprocessed"
    return os.path.join(cfg["paths"]["ocr_cache"], f"{tag}.json")
