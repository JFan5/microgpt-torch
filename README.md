# MyGPT

A minimal GPT implementation from scratch in pure Python, for educational purposes. Inspired by [Andrej Karpathy's makemore](https://github.com/karpathy/makemore).

## Overview

This project implements a small GPT (Generative Pre-trained Transformer) language model using only Python standard libraries — no PyTorch, no NumPy. It includes:

- **Autograd engine**: A `Value` class that supports automatic differentiation via reverse-mode backpropagation
- **Transformer architecture**: Token/position embeddings, multi-head self-attention, MLP blocks, RMSNorm, and residual connections
- **Training loop**: Epoch-based training with Adam optimizer and linear learning rate decay
- **Text generation**: Temperature-controlled sampling to generate new names

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
python MyGPT.py
```

The script will:
1. Download the dataset (if not present)
2. Train the model for 40 epochs
3. Generate 20 new "hallucinated" names

## How It Works

1. **Tokenization**: Each character is mapped to an integer ID (a=0, b=1, ..., z=25, BOS=26)
2. **Forward pass**: For each token, compute embeddings -> multi-head attention -> MLP -> logits
3. **Loss**: Cross-entropy loss (negative log probability of the target token)
4. **Backward pass**: Automatic differentiation computes gradients for all parameters
5. **Update**: Adam optimizer updates parameters with linear learning rate decay
6. **Generation**: Sample tokens autoregressively with temperature-controlled softmax
