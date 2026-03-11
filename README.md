# MicroGpt-torch

A minimal GPT implementation in Python + PyTorch, for educational purposes.

This is the PyTorch + CUDA rewrite of [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) by Andrej Karpathy, which is a pure Python implementation using a custom `Value` class for automatic differentiation. This version replaces the scalar-level autograd with PyTorch tensor operations for significantly faster training on GPU.

## Overview

This project implements a small GPT (Generative Pre-trained Transformer) language model in a single ~120-line Python file (excluding comments). It uses PyTorch for tensor operations and CUDA acceleration. It includes:

- **Transformer architecture**: Token/position embeddings, multi-head self-attention with causal mask, MLP blocks, RMSNorm, and residual connections
- **Training loop**: Epoch-based training with `torch.optim.Adam` and linear learning rate decay
- **Text generation**: Temperature-controlled sampling with `torch.multinomial`
- **CUDA support**: Automatically uses GPU when available

## Model Architecture

| Component      | Config         |
| -------------- | -------------- |
| Embedding dim  | 16             |
| Attention heads | 4             |
| Transformer layers | 1          |
| Block size     | 16             |
| Vocab size     | 27 (a-z + BOS) |

## Dataset

The model trains on ~32K English names from [names.txt](https://github.com/karpathy/makemore/blob/master/names.txt). Each name is treated as a document, and each character as a token.

## Usage

```bash
conda run -n llmstl python microgpt-torch.py
```

The script will:
1. Download the dataset (if not present)
2. Train the model for 40 epochs on GPU
3. Generate 20 new "hallucinated" names

## How It Works

1. **Tokenization**: Each character is mapped to an integer ID (a=0, b=1, ..., z=25, BOS=26)
2. **Forward pass**: Full sequence processed at once — embeddings -> multi-head attention (with causal mask) -> MLP -> logits
3. **Loss**: `F.cross_entropy` over the entire sequence
4. **Backward pass**: `loss.backward()` via PyTorch autograd
5. **Update**: `torch.optim.Adam` with `LambdaLR` linear decay
6. **Generation**: Autoregressive sampling under `torch.no_grad()` with temperature-controlled softmax
