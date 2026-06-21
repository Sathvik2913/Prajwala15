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
