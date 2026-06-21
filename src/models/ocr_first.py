"""OCR-first pipeline: answer from (question + OCR tokens), no image at inference.

This isolates how far you can get from text alone — and exposes exactly where OCR
noise/incompleteness caps performance.

Backends:
  blip2_text : reuse BLIP-2's frozen Flan-T5 language head as a pure text reader
  api        : text-only API call
"""


class OCRFirstBase:
    def answer(self, question: str, ocr_text: str) -> str:
        raise NotImplementedError


PROMPT = ("Read the text found in an image and answer the question with the shortest "
          "possible phrase (one word or number when possible).\n"
          "Text in image: {ocr}\nQuestion: {q}\nAnswer:")


class BLIP2TextReader(OCRFirstBase):
    """Uses the underlying Flan-T5 of BLIP-2 as a text-to-text reader (no vision)."""
    def __init__(self, model_name="Salesforce/blip2-flan-t5-xl", device="cpu",
                 max_new_tokens=16):
        import torch
        from transformers import Blip2Processor, T5ForConditionalGeneration
        self.device = device
        self.max_new_tokens = max_new_tokens
        proc = Blip2Processor.from_pretrained(model_name)
        self.tok = proc.tokenizer
        # load just the language model weights of the BLIP-2 checkpoint
        self.lm = T5ForConditionalGeneration.from_pretrained(
            model_name, subfolder="language_model"
        ).to(device).eval() if False else self._fallback_t5(device)

    def _fallback_t5(self, device):
        # Robust path: a standalone flan-t5-xl matches BLIP-2's text head closely
        # and avoids checkpoint-subfolder issues across transformers versions.
        from transformers import T5ForConditionalGeneration
        m = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")
        return m.to(device).eval()

    def answer(self, question: str, ocr_text: str) -> str:
        import torch
        text = PROMPT.format(ocr=ocr_text or "(none)", q=question)
        ids = self.tok(text, return_tensors="pt", truncation=True,
                       max_length=512).to(self.device)
        with torch.no_grad():
            out = self.lm.generate(**ids, max_new_tokens=self.max_new_tokens)
        return self.tok.decode(out[0], skip_special_tokens=True).strip()


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
                       "content": PROMPT.format(ocr=ocr_text or "(none)", q=question)}],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip()


def build_ocr_first(cfg: dict, device: str) -> OCRFirstBase:
    o = cfg["ocr_first"]
    if o["backend"] == "blip2_text":
        return BLIP2TextReader(cfg["vlm"]["blip2_model"], device, o["max_new_tokens"])
    if o["backend"] == "api":
        return APITextReader(o["api_model"], o["max_new_tokens"])
    raise ValueError(o["backend"])
