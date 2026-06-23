"""OCR-first pipeline: answer from (question + OCR tokens), no image at inference."""
from ..fusion import build_fusion_prompt


class OCRFirstBase:
    def answer(self, question: str, ocr_text: str) -> str:
        raise NotImplementedError


class TextQAModel(OCRFirstBase):
    """OCR tokens + question → text-to-text model (Flan-T5 / similar)."""

    def __init__(self, model_name="google/flan-t5-base", device="cpu", max_new_tokens=16):
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tok = T5Tokenizer.from_pretrained(model_name)
        self.lm = T5ForConditionalGeneration.from_pretrained(model_name).to(device).eval()
        self.model_name = model_name

    def answer(self, question: str, ocr_text: str) -> str:
        import torch
        text = build_fusion_prompt(question, ocr_text)
        ids = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=512).to(self.device)
        with torch.no_grad():
            out = self.lm.generate(**ids, max_new_tokens=self.max_new_tokens)
        return self.tok.decode(out[0], skip_special_tokens=True).strip()


# Backward-compatible alias
BLIP2TextReader = TextQAModel


class APITextReader(OCRFirstBase):
    def __init__(self, model="claude-sonnet-4-6", max_new_tokens=16):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max(32, max_new_tokens)

    def answer(self, question: str, ocr_text: str) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system="You are given OCR text extracted from an image and a question. "
                   "Answer with the shortest phrase, usually one word or number. "
                   "If the OCR text does not contain the answer, give your best guess.",
            messages=[{"role": "user",
                       "content": build_fusion_prompt(question, ocr_text)}],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip()


def build_ocr_first(cfg: dict, device: str) -> OCRFirstBase:
    o = cfg["ocr_first"]
    if o["backend"] in ("blip2_text", "text_qa"):
        model = o.get("text_model", "google/flan-t5-base")
        return TextQAModel(model, device, o["max_new_tokens"])
    if o["backend"] == "api":
        return APITextReader(o["api_model"], o["max_new_tokens"])
    raise ValueError(o["backend"])
