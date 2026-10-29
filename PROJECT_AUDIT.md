# DeepGuard Project Audit & Technical Baseline Report

**Date:** 2026-09-16  
**Auditor:** Antigravity AI Engineering Engine  
**Target:** DeepGuard / Nivora Deepfake Detection Subsystem Refactor & Production Blueprint

---

## 1. Executive Summary

A comprehensive architectural and codebase audit of `DeepGuard` was conducted to transition the existing hackathon prototype into a modular, production-ready **DeepQShield** deepfake detection engine integrated with Post-Quantum Cryptography (PQC) and live video temporal risk analysis.

---

## 2. Directory & Component Inventory

| Path | Category | Status / Description |
| :--- | :--- | :--- |
| `backend/app/main.py` | API Entrypoint | Functional FastAPI app with CORS middleware & exception handler. |
| `backend/app/ml/deepfake_model.py` | Model Adapter | **Legacy/Heuristic.** Uses OpenCV Haar cascades + FFT spectral ratio + chrominance variance heuristics. Contains hardcoded demo adapters. |
| `backend/app/services/deepfake_service.py` | Business Logic | Functional frame processing wrapper. Delegates to `LightweightMLAdapter` or `DemoModelAdapter`. |
| `backend/app/routes/verification.py` | API Route | Contains `/api/auth/deepfake` endpoint. |
| `frontend/` | UI Dashboard | React + Vite + Tailwind CSS dashboard with WebRTC stream capture & simulation controls. |
| `models/` | Model Weights | Contains `README.md` only. No pre-trained or trained PyTorch DeepQShield checkpoints present. |
| `datasets/` | Data Storage | Folder structures present (`genuine`, `deepfake`, `replay`), but lacks automated DFDC loader & dataset pipeline. |

---

## 3. Working, Broken, and Missing Features

### Working Features
- FastAPI Web Server infrastructure & routing.
- React frontend WebRTC webcam capture and risk display UI.
- Base64 image payload to OpenCV array decoding (`backend/app/utils/files.py`).
- Basic test suite (`backend/tests/test_api.py`, `backend/tests/test_risk_engine.py`) passing with proper `PYTHONPATH`.

### Broken / Prototype Limitations
- Deepfake detection is using simple heuristic FFT/chrominance thresholds instead of the DeepQShield paper's neural model.
- Demo mode relies on hardcoded probability scores (`0.93`, `0.72`, `0.48`, `0.06`).
- Absence of real model training, validation, checkpointing, and metrics pipeline.
- Absence of adversarial robustness testing (FGSM, BIM, PGD).
- Absence of Post-Quantum Cryptographic signing (Kyber key encapsulation, Dilithium signatures, SHA3 hashing).

### Missing Core Features (Required by DeepQShield Paper & Production Specification)
1. **ResNeXt50_32×4d Backbone:** PyTorch feature extractor backbone.
2. **LWE / Lattice-Based Feature Enhancement:** Differentiable lattice-inspired perturbation & enhancement module.
3. **Feature Fusion:** Fusion layer combining ResNeXt features + LWE enhanced features with shape validation.
4. **Attention Mechanism:** Spatial/channel attention layer.
5. **Classifier Head:** Dense → ReLU → Dropout (0.3) → Real/Fake probability output.
6. **Training Pipeline:** Focal Loss + Label Smoothing + AdamW + CosineAnnealingWarmRestarts scheduler + DFDC dataset loader.
7. **Adversarial Engine:** FGSM, BIM, and PGD attack evaluation comparing Baseline ResNeXt vs DeepQShield.
8. **Live Video Processing & Temporal Aggregation:** Frame sampling, face tracking, temporal aggregation, suspicious interval detection.
9. **FastAPI Deepfake Microservice (v1 API):** `/api/v1/model/info`, `/api/v1/deepfake/image`, `/api/v1/deepfake/video`, `/api/v1/deepfake/live/*`.
10. **PQC Module (`security/pqc`):** Kyber key exchange, Dilithium digital signature signing/verification, and SHA3-256 result hashing.

---

## 4. Dependency & Environment Analysis

- **Python Version:** 3.13.2 (Windows OS).
- **Core ML Framework:** PyTorch & Torchvision being provisioned.
- **Crypto Libraries:** PyCryptodome / Cryptography / isolated PQC provider abstraction.
- **Utility Tools:** OpenCV, SciPy, Pillow, NumPy, PyYAML, Scikit-Learn, Matplotlib, Pandas, Albumentations.

---

## 5. Recommended Migration Plan

1. **Environment Setup & Core Architecture**:
   - Establish clean PyTorch ML environment and folder structure (`models/`, `video/`, `training/`, `adversarial/`, `security/pqc/`, `configs/`, `tests/`).
2. **Modular Model Engineering**:
   - Implement `ResNeXt50_32x4d`, `LWELayer`, `FeatureFusion`, `Attention`, `Classifier`, and `DeepQShieldModel`.
3. **Training & Loss Engine**:
   - Implement `FocalLoss`, label smoothing, AdamW, Cosine scheduler, and dataset loader.
4. **Adversarial & Ablation Framework**:
   - Implement FGSM, BIM, PGD attacks & automated ablation script comparing baseline vs robust model.
5. **Video Pipeline & Temporal Aggregation**:
   - Build frame sampler, face detector, temporal aggregator, and suspicious interval detection.
6. **FastAPI Services & Live Analysis Session**:
   - Build REST endpoints under `/api/v1/deepfake/` and connect to business logic.
7. **Post-Quantum Cryptography (PQC) & Integrity**:
   - Implement SHA3-256 hashing, Dilithium signature signing, and Kyber key establishment.
8. **Frontend Integration & Demo Studio**:
   - Update UI to consume live frame analysis and present PQC-verified evidence timeline.
9. **Verification, Performance Benchmarks & Documentation**:
   - Unit tests, integration tests, latency benchmarks, `RESEARCH_MAPPING.md`, `RESEARCH_REPRODUCTION.md`, `SECURITY.md`, and Docker files.
