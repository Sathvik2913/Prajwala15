"""OCR + question fusion for Member 1.

Concatenates OCR tokens with the question and formats a prompt for a text QA model.
"""

FUSION_STYLE = "prompt"  # prompt | concat


def build_fusion_prompt(question: str, ocr_text: str, style: str = FUSION_STYLE) -> str:
    ocr_text = ocr_text.strip() if ocr_text else "(none)"
    if style == "concat":
        return f"Question: {question} OCR: {ocr_text}"
    return (
        "Read the text found in an image and answer the question with the shortest "
        "possible phrase (one word or number when possible).\n"
        f"Text in image: {ocr_text}\n"
        f"Question: {question}\n"
        "Answer:"
    )
