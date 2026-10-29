# DeepQShield Neural Network Model Specification

This document defines the mathematical architecture and module specifications for **DeepQShield**.

---

## 1. Network Architecture

```text
Input Frame (3 x 224 x 224)
         │
ResNeXt50_32x4d Backbone
         │ (2048-dim)
   ┌─────┴─────┐
   │           │
   ▼           ▼
Backbone    LWE Lattice Layer (2048-dim)
   │           │
   └─────┬─────┘
         ▼
    Feature Fusion (2048-dim)
         │
         ▼
     Attention (2048-dim)
         │
         ▼
  Classifier Head (Dense -> ReLU -> Dropout=0.3 -> Output)
         │
         ▼
   Real / Fake Logits
```

---

## 2. Parameter Summary

- **Backbone Output:** 2048 dimensions
- **Lattice Dimensions:** 2048
- **Dropout Rate:** 0.3
- **Classification Output:** 2 classes (0: REAL, 1: DEEPFAKE)
