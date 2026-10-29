# Adversarial Robustness Testing Guide

This guide details how to evaluate DeepQShield model resilience against gradient-based adversarial attacks: **FGSM**, **BIM**, and **PGD**.

---

## 1. Attack Algorithms

1. **Fast Gradient Sign Method (FGSM):** One-step gradient sign perturbation.
2. **Basic Iterative Method (BIM):** Multi-step iterative FGSM bounded within $\epsilon$.
3. **Projected Gradient Descent (PGD):** Multi-step gradient descent with random initialization inside $\epsilon$-ball.

---

## 2. Running Adversarial Evaluation Benchmark

Run the automated adversarial benchmark script:

```bash
python scripts/run_adversarial_eval.py
```

This generates `reports/adversarial_results.csv` comparing clean vs perturbed accuracy across models:

- Baseline ResNeXt50
- Full DeepQShield (ResNeXt + LWE + Fusion + Attention)
