# DeepQShield Model Training Guide

This guide details how to execute training, validation, and checkpoint saving for the **DeepQShield** neural model.

---

## 1. Quick Start Training

To launch training using default hyperparameters (AdamW, Cosine Scheduler, Focal Loss):

```bash
python training/train.py
```

---

## 2. Configurable Hyperparameters

Edit `configs/deepqshield.yaml` to modify training parameters:

```yaml
training:
  batch_size: 32
  learning_rate: 0.0001
  weight_decay: 0.0001
  epochs: 50
  focal_loss_gamma: 2.0
  focal_loss_alpha: 0.25
  label_smoothing: 0.1
  optimizer: "AdamW"
  scheduler: "CosineAnnealingWarmRestarts"
```

---

## 3. Checkpointing & Outputs

- Checkpoints are saved to `checkpoints/deepqshield_best.pth`.
- Training history logs metrics: Loss, Accuracy, F1, and ROC-AUC.
