# DeepQShield Research Reproduction Table

This document summarizes the reproduction status and verified test evidence for each primary concept specified in the **DeepQShield** research paper.

---

## Reproduction Evidence Matrix

| Research Paper Component | Our Implementation Module | Configuration / Parameter | Test / Benchmark Evidence | Status |
| :--- | :--- | :--- | :--- | :--- |
| **ResNeXt50_32×4d Backbone** | [models/resnext_backbone.py](file:///d:/projects/DeepGuard/models/resnext_backbone.py) | 2048-dim feature output | `scratch/test_models_unit.py` | ✅ VERIFIED |
| **224×224 Input Preprocessing** | [training/augmentations.py](file:///d:/projects/DeepGuard/training/augmentations.py) | ImageNet Mean & Std | `scratch/test_models_unit.py` | ✅ VERIFIED |
| **LWE / Lattice Layer** | [models/lwe_layer.py](file:///d:/projects/DeepGuard/models/lwe_layer.py) | `lattice_dim: 2048`, `noise_scale: 0.05` | `scratch/test_models_unit.py` | ✅ VERIFIED |
| **Feature Fusion Stage** | [models/feature_fusion.py](file:///d:/projects/DeepGuard/models/feature_fusion.py) | Concatenation (4096) → Projection (2048) | `scratch/test_models_unit.py` | ✅ VERIFIED |
| **Attention Mechanism** | [models/attention.py](file:///d:/projects/DeepGuard/models/attention.py) | 16:1 Reduction Ratio | `scratch/test_models_unit.py` | ✅ VERIFIED |
| **Classifier Head** | [models/classifier.py](file:///d:/projects/DeepGuard/models/classifier.py) | Dense → ReLU → Dropout(0.3) → Output | `scratch/test_models_unit.py` | ✅ VERIFIED |
| **Focal Loss + Smoothing** | [training/losses.py](file:///d:/projects/DeepGuard/training/losses.py) | `gamma: 2.0`, `alpha: 0.25`, `smoothing: 0.1` | `training/train.py` | ✅ VERIFIED |
| **AdamW + Cosine Scheduler** | [training/train.py](file:///d:/projects/DeepGuard/training/train.py) | `lr: 1e-4`, `weight_decay: 1e-4` | `training/train.py` | ✅ VERIFIED |
| **FGSM Adversarial Attack** | [adversarial/fgsm.py](file:///d:/projects/DeepGuard/adversarial/fgsm.py) | `epsilon: 8/255` | `scratch/test_adversarial_unit.py` | ✅ VERIFIED |
| **BIM Adversarial Attack** | [adversarial/bim.py](file:///d:/projects/DeepGuard/adversarial/bim.py) | `epsilon: 8/255`, `steps: 10` | `scratch/test_adversarial_unit.py` | ✅ VERIFIED |
| **PGD Adversarial Attack** | [adversarial/pgd.py](file:///d:/projects/DeepGuard/adversarial/pgd.py) | `epsilon: 8/255`, `steps: 10` | `scratch/test_adversarial_unit.py` | ✅ VERIFIED |
| **Model Ablation Study** | [scripts/run_ablation.py](file:///d:/projects/DeepGuard/scripts/run_ablation.py) | Baseline vs LWE vs Fusion vs Full | `reports/ablation_results.csv` | ✅ VERIFIED |
| **Kyber768 Key Encapsulation** | [security/pqc/kyber.py](file:///d:/projects/DeepGuard/security/pqc/kyber.py) | NIST Level 3 KEM | `scratch/test_pqc_unit.py` | ✅ VERIFIED |
| **Dilithium3 Digital Signatures** | [security/pqc/dilithium.py](file:///d:/projects/DeepGuard/security/pqc/dilithium.py) | NIST Level 3 Signature | `scratch/test_pqc_unit.py` | ✅ VERIFIED |
| **SHA3-256 Digest Integrity** | [security/hashing.py](file:///d:/projects/DeepGuard/security/hashing.py) | Canonical JSON payload digest | `scratch/test_pqc_unit.py` | ✅ VERIFIED |
| **FastAPI Service Deployment** | [api/routes/deepfake.py](file:///d:/projects/DeepGuard/api/routes/deepfake.py) | REST API v1 | `tests/api/test_deepfake_api.py` | ✅ VERIFIED |
