# Reading Text in Images for Visual Question Answering

Compare **OCR-first** and **vision-language (VLM)** approaches on **TextVQA**, with
analysis of failure cases where OCR is noisy or incomplete.

## What this project does

1. **OCR-first pipeline** — extract text tokens from each image (Tesseract / EasyOCR),
   feed `[question + OCR tokens]` to a text reasoner, and answer.
2. **VLM pipeline** — answer directly from `[image + question]` using a frozen
   multimodal model (BLIP-2 / a vision-language API).
3. **Hybrid pipeline** — give the VLM the OCR tokens as an auxiliary hint.
4. **Evaluation** — official TextVQA accuracy (soft-voting over 10 human answers).
5. **Error analysis** — bucket results by OCR quality to show *where* and *why* each
   approach wins/loses (the core point of the project).

```
textvqa/
├── README.md
├── requirements.txt
├── configs/default.yaml
├── data/                      # you put TextVQA json + images here
├── src/
│   ├── dataset.py             # TextVQA loader
│   ├── ocr.py                 # Tesseract / EasyOCR wrappers
│   ├── normalize.py          # official answer normalization
│   ├── metrics.py             # TextVQA soft accuracy + ANLS
│   ├── models/
│   │   ├── ocr_first.py       # OCR -> text reasoner
│   │   ├── vlm.py             # frozen VLM (BLIP-2 / API)
│   │   └── hybrid.py          # VLM + OCR hint
│   └── ocr_quality.py         # OCR-vs-GT-answer coverage score for analysis
├── scripts/
│   ├── 00_download.md         # how to get the data (network-gated)
│   ├── 01_run_ocr.py          # cache OCR for all images
│   ├── 02_eval.py             # run a pipeline + score it
│   └── 03_analysis.py         # buckets, plots, comparison table
└── outputs/                   # predictions, scores, figures
```

## Quick start (Colab)

Open **`Reading_Text_in_Images.ipynb`** in Google Colab (GPU runtime). All team members run their sections in that single notebook:

| Section | Member | Scripts |
|---------|--------|---------|
| OCR + OCR-first baseline + error analysis | Member 1 | `01_run_ocr.py`, `02_eval.py`, `04_ocr_error_analysis.py` |
| VLM (BLIP-2) | Member 2 | `02_eval.py --approach vlm` |
| Hybrid + comparison | Member 3 | `02_eval.py --approach hybrid`, `03_analysis.py` |

Local CLI equivalent:

```bash
pip install -r requirements.txt
# 1. download data (see scripts/00_download.md), put under data/
# 2. cache OCR once (Member 1):
python scripts/01_run_ocr.py --split val --engine easyocr
python scripts/01_run_ocr.py --split val --engine easyocr --preprocess
# 3. evaluate approaches:
python scripts/02_eval.py --approach ocr_first --split val
python scripts/04_ocr_error_analysis.py --split val
python scripts/02_eval.py --approach vlm       --split val
python scripts/02_eval.py --approach hybrid    --split val
python scripts/03_analysis.py --split val
```

## Why three pipelines

The interesting result is not a single accuracy number — it's the **interaction
between OCR quality and answer source**. `03_analysis.py` produces a table like:

| OCR coverage of GT answer | OCR-first acc | VLM acc | Hybrid acc |
|---|---|---|---|
| full (answer token present) | high | mid | high |
| partial | mid | mid | high |
| none (OCR missed it) | low | depends on VLM's own reading | mid |

This is exactly the "noisy/incomplete OCR" comparison the brief asks for.

## Notes on the environment
Downloading TextVQA (~25 GB) and running OCR/VLMs needs a machine with internet and
ideally a GPU. The code is written to run on CPU for small `--limit` smoke tests and
to scale up on GPU for the full split.
