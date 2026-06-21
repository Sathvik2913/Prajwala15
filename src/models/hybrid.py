"""Hybrid pipeline: give the VLM the image AND the OCR tokens as a hint.

This tests whether OCR helps a VLM that already "reads" — the practically strongest
setup, and the one that should be most robust to partial OCR.
"""


class HybridBase:
    def answer(self, image, question: str, ocr_text: str) -> str:
        raise NotImplementedError


HINT = "Text spotted in the image (may be noisy): {ocr}"


class BLIP2Hybrid(HybridBase):
    def __init__(self, model_name="Salesforce/blip2-flan-t5-xl", device="cpu",
                 max_new_tokens=16):
        from .vlm import BLIP2VLM
        self.vlm = BLIP2VLM(model_name, device, max_new_tokens)

    def answer(self, image, question: str, ocr_text: str) -> str:
        q = f"{HINT.format(ocr=ocr_text or '(none)')}. Question: {question}"
        # reuse the VLM prompt builder via answer()
        return self.vlm.answer_batch([image], [q])[0]


class APIHybrid(HybridBase):
    def __init__(self, model="claude-sonnet-4-6", max_new_tokens=16):
        import anthropic
        from .vlm import APIVLM
        self.api = APIVLM(model, max_new_tokens)

    def answer(self, image, question: str, ocr_text: str) -> str:
        q = f"{HINT.format(ocr=ocr_text or '(none)')}\nQuestion: {question}"
        return self.api.answer(image, q)


def build_hybrid(cfg: dict, device: str) -> HybridBase:
    v = cfg["vlm"]
    if v["backend"] == "blip2":
        return BLIP2Hybrid(v["blip2_model"], device, v["max_new_tokens"])
    if v["backend"] == "api":
        return APIHybrid(v["api_model"], v["max_new_tokens"])
    raise ValueError(v["backend"])
