# Self-Pruning Neural Network — Analysis Report

## Why L1 Penalty on Sigmoid Gates Encourages Sparsity?

Each PrunableLinear layer has a learnable `gate_scores` tensor.
During forward pass, Sigmoid converts scores to gates between 0 and 1:

gates = sigmoid(gate_scores)
pruned_weights = weight × gates

**Sparsity Loss** is the L1 norm of all gates across all layers:

SparsityLoss = Σ sigmoid(gate_scores)
TotalLoss = ClassificationLoss + λ × SparsityLoss

## Why L1 and not L2?

- **L1 norm** penalizes every non-zero value equally → pushes values to **exactly zero** → true sparsity
- **L2 norm** penalizes large values more but allows small non-zero values → no true sparsity

## Results Table

| Lambda (λ) | Test Accuracy | Sparsity Level |
|------------|---------------|----------------|
| 0.00001    | 60.69%        | 87.82%         |
| 0.00100    | 60.40%        | 99.98%         |
| 0.10000    | 57.70%        | 100.00%        |

## Analysis of λ Trade-off

### λ = 1e-5 (Low)
- Sparsity penalty is small, network focuses mostly on classification
- Result: **60.69% accuracy** with **87.82% sparsity**

### λ = 1e-3 (Medium) — Best Model 
- Balanced trade-off between accuracy and sparsity
- Result: **60.40% accuracy** with **99.98% sparsity** — the sweet spot

### λ = 0.1 (High)
- Sparsity penalty dominates, forces almost all gates to zero
- Result: **57.70% accuracy** with **100% sparsity** — over-pruned

## Gate Distribution

- **Low λ**: Gates spread between 0 and 1, moderate spike near 0
- **Medium λ**: Large spike at 0, very few gates active — confirms 99.98% sparsity
- **High λ**: Almost all gates collapsed to 0 — fully pruned

## Conclusion

The best model (λ = 1e-3) achieves **99.98% sparsity with only 0.29% accuracy drop**,
demonstrating that most weights are redundant for CIFAR-10 classification.
