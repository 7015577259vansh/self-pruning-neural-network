# Self-Pruning Neural Network on CIFAR-10

**Case Study — AI Engineer | Tredence Analytics**

---

## Overview

This project implements a **Self-Pruning Neural Network** that learns to remove its
own unnecessary weights during training — without any post-training pruning step.

Instead of pruning after training, the network uses a custom **gated linear layer**
and a **sparsity regularization loss** to dynamically prune itself on the fly.

---

## How It Works

### PrunableLinear Layer
Each linear layer has an extra learnable `gate_scores` tensor (same shape as weights).

```
gates        = sigmoid(gate_scores)       # values between 0 and 1
pruned_weight = weight × gates            # element-wise multiply
output        = pruned_weight @ x + bias  # standard linear op
```

When a gate approaches 0, it effectively **removes** that weight from the network.

### Sparsity Loss
```
TotalLoss = CrossEntropyLoss + λ × SparsityLoss
SparsityLoss = sum of all gate values (L1 norm)
```

The L1 penalty pushes gates toward exactly zero, creating true sparsity.

---

## Results

| Lambda (λ) | Test Accuracy | Sparsity |
|:----------:|:-------------:|:--------:|
| 1e-5       | 60.69%        | 87.82%   |
| 1e-3       | 60.40%        | 99.98%   |
| 0.1        | 57.70%        | 100.00%  |

**Best model: λ = 1e-3** — 99.98% sparsity with only 0.29% accuracy drop.

---

## Files

```
self-pruning-neural-network/
├── trendenct.ipynb          ← Main notebook (training + evaluation)
├── report.md                ← Analysis report with results
├── outputs/
│   └── gate_distributions.png  ← Gate histogram plots
└── README.md
```

---

## Tech Stack

- Python 3.12
- PyTorch
- torchvision (CIFAR-10 dataset)
- matplotlib
- Google Colab (T4 GPU)

---

## Key Takeaways

- L1 penalty on sigmoid gates effectively encourages sparsity
- The network learns **which weights matter** autonomously
- λ controls the sparsity-accuracy trade-off cleanly
- 99.98% of weights can be pruned with negligible accuracy loss
