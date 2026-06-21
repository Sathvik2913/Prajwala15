"""Measure how well OCR covered the *ground-truth answer* for each sample.

This is the axis the project cares about: when OCR is complete vs noisy/incomplete,
which approach wins? We bucket every sample by `coverage`:

  full    : every GT-answer token appears among OCR tokens
  partial : some but not all GT-answer tokens appear
  none    : no GT-answer token appears in OCR

GT answer used = the most common normalized human answer for that question.
"""
from collections import Counter

from .normalize import normalize_answer


def _majority_answer(answers: list[str]) -> str:
    if not answers:
        return ""
    norm = [normalize_answer(a) for a in answers if a.strip()]
    if not norm:
        return ""
    return Counter(norm).most_common(1)[0][0]


def coverage_bucket(ocr_text: str, gt_answers: list[str]) -> str:
    gt = _majority_answer(gt_answers)
    if not gt:
        return "unknown"
    gt_tokens = set(gt.split())
    ocr_tokens = set(normalize_answer(ocr_text).split())
    hit = gt_tokens & ocr_tokens
    if not hit:
        return "none"
    if hit == gt_tokens:
        return "full"
    return "partial"
