# FlashAlopex (CUDA-only)

FlashAlopex: a fused CUDA implementation of the Alopex optimizer (stochastic correlation-based updates).
This repo provides a CUDA kernel + PyTorch extension + experiments (XOR, MNIST Logistic, CIFAR-10 CNN).

## Quickstart

Requirements:
- Linux with CUDA toolkit matching your PyTorch CUDA build
- Python 3.9+
- PyTorch with CUDA