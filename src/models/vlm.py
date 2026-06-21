"""Vision-language pipeline: answer directly from (image, question).

Two backends:
  blip2 : frozen Salesforce/blip2-flan-t5-xl (local, GPU recommended)
  api   : a vision-language API (sends image + question)
"""
import base64
import io


class VLMBase:
    def answer(self, image, question: str) -> str:
        raise NotImplementedError

    def answer_batch(self, images, questions):
        return [self.answer(im, q) for im, q in zip(images, questions)]


class BLIP2VLM(VLMBase):
    def __init__(self, model_name="Salesforce/blip2-flan-t5-xl", device="cpu",
                 max_new_tokens=16):
        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        self.device = device
        self.max_new_tokens = max_new_tokens
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.proc = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(device)
        self.model.eval()

    def _prompt(self, q: str) -> str:
        return f"Question: {q} Answer:"

    def answer(self, image, question: str) -> str:
        return self.answer_batch([image], [question])[0]

    def answer_batch(self, images, questions):
        import torch
        prompts = [self._prompt(q) for q in questions]
        inputs = self.proc(images=list(images), text=prompts,
                           return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        texts = self.proc.batch_decode(out, skip_special_tokens=True)
        return [t.strip() for t in texts]


class APIVLM(VLMBase):
    """Vision-language answering via the Anthropic Messages API."""
    def __init__(self, model="claude-sonnet-4-6", max_new_tokens=16):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max(32, max_new_tokens)

    @staticmethod
    def _b64(image):
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        return base64.standard_b64encode(buf.getvalue()).decode()

    def answer(self, image, question: str) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system="Answer the visual question with the shortest possible phrase, "
                   "usually a single word or number read from the image. No punctuation.",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                     "media_type": "image/jpeg", "data": self._b64(image)}},
                    {"type": "text", "text": question},
                ],
            }],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip()


def build_vlm(cfg: dict, device: str) -> VLMBase:
    v = cfg["vlm"]
    if v["backend"] == "blip2":
        return BLIP2VLM(v["blip2_model"], device, v["max_new_tokens"])
    if v["backend"] == "api":
        return APIVLM(v["api_model"], v["max_new_tokens"])
    raise ValueError(v["backend"])
