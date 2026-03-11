# MicroGpt-torch

A minimal GPT implementation in **83 lines** of Python + PyTorch. Rewrite of [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) by Andrej Karpathy, replacing scalar-level autograd with PyTorch tensor ops + CUDA.

Trains on ~32K English names and generates new ones.

## Requirements

- Python >= 3.10
- PyTorch >= 2.4 (for `F.rms_norm`, `F.scaled_dot_product_attention`)

## Usage

```bash
python microgpt-torch.py
```

The script will:
1. Download the dataset (if not present)
2. Train for 40 epochs (auto-detects GPU)
3. Generate 20 new "hallucinated" names

## Files

- `microgpt-torch.py` — compact version (83 lines)
- `microgpt-torch-comments.py` — same code with bilingual comments (EN/CN)
