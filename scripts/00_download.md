# Getting the TextVQA data

The sandbox has no internet for these hosts, so download on your own machine.

## Annotations (small)
From https://textvqa.org/dataset/ :
- `TextVQA_0.5.1_train.json`
- `TextVQA_0.5.1_val.json`
- `TextVQA_0.5.1_test.json`

Put them in `data/`.

## Images (~25 GB)
TextVQA images come from OpenImages. The dataset page provides:
- `train_val_images.zip`  -> unzip to `data/train_images/`
- `test_images.zip`       -> unzip to `data/test_images/`

(Val images are inside the train_val set; the loader resolves paths by `image_id`.)

## Alternative: Hugging Face
```python
from datasets import load_dataset
ds = load_dataset("textvqa")          # has images + questions + answers
```
If you use HF instead of the raw json, swap `src/dataset.py`'s loader for one that
iterates the HF split — the rest of the pipeline only needs each sample to expose
`question_id, question, image (PIL), answers (list[str])`.

## Sanity check
```bash
python -c "from src.dataset import TextVQADataset; \
d=TextVQADataset('val'); print(len(d)); print(d[0]['question'])"
```
