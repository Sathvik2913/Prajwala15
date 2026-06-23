"""Image preprocessing for noisy scene text (Member 1).

Steps: morphological denoise on luminance → mild contrast → optional deskew.
Output stays RGB for Tesseract, EasyOCR, and PaddleOCR.
"""
from PIL import Image
import numpy as np


def _estimate_skew_angle(gray: np.ndarray) -> float:
    import cv2

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return float(angle)


def _rotate_rgb(rgb: np.ndarray, angle: float) -> np.ndarray:
    import cv2

    if abs(angle) < 0.5:
        return rgb
    h, w = rgb.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        rgb, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def preprocess_image(
    image: Image.Image,
    denoise: bool = True,
    enhance: bool = True,
    deskew: bool = True,
) -> Image.Image:
    """Return an RGB PIL image after preprocessing."""
    import cv2

    rgb = np.array(image.convert("RGB"))
    if denoise:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        lab[:, :, 0] = cv2.morphologyEx(lab[:, :, 0], cv2.MORPH_OPEN, kernel)
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if enhance:
        rgb = cv2.convertScaleAbs(rgb, alpha=1.15, beta=8)
    if deskew:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        angle = _estimate_skew_angle(gray)
        if abs(angle) >= 1.0:
            rgb = _rotate_rgb(rgb, angle)
    return Image.fromarray(rgb)
