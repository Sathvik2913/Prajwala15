"""TextVQA soft accuracy + ANLS."""
from .normalize import normalize_answer


def vqa_accuracy(pred: str, gt_answers: list[str]) -> float:
    """Official soft accuracy: min(#matching/3, 1), averaged over the 10 leave-one-out
    subsets. Equivalent closed form below."""
    if not gt_answers:
        return 0.0
    p = normalize_answer(pred)
    gts = [normalize_answer(a) for a in gt_answers]
    accs = []
    for i in range(len(gts)):
        others = gts[:i] + gts[i + 1:]
        matches = sum(1 for o in others if o == p)
        accs.append(min(matches / 3.0, 1.0))
    return sum(accs) / len(accs)


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred: str, gt_answers: list[str], tau: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity (used in ST-VQA-style scoring)."""
    if not gt_answers:
        return 0.0
    p = normalize_answer(pred)
    best = 0.0
    for g in gt_answers:
        g = normalize_answer(g)
        if not p and not g:
            nl = 0.0
        else:
            nl = _levenshtein(p, g) / max(len(p), len(g), 1)
        s = 1 - nl
        best = max(best, s if s >= tau else 0.0)
    return best


def aggregate(rows: list[dict]) -> dict:
    n = len(rows) or 1
    return {
        "n": len(rows),
        "vqa_accuracy": sum(r["vqa_acc"] for r in rows) / n,
        "anls": sum(r["anls"] for r in rows) / n,
    }
