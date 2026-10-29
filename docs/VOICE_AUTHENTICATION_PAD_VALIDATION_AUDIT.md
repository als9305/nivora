# Module 4 — Voice Authentication & Audio Anti-Spoofing
## Real Audio Presentation Attack Detection (PAD) Scientific Validation Audit Report

**System Name:** Nivora DeepGuard Security Platform  
**Module Under Audit:** Module 4 — Voice Authentication & Audio Anti-Spoofing  
**Audit Type:** Real Audio Presentation Attack Detection (PAD) Scientific Validation  
**Date:** September 20, 2026  
**Module Status:** **UNLOCKED** (Software Implementation Verified; Biometric Generalization Pending)  

---

### Executive Summary

This report presents the scientific validation audit for **Audio Presentation Attack Detection (PAD)** in **Module 4 (Voice Authentication & Audio Anti-Spoofing)** of the Nivora DeepGuard platform.

To replace randomly initialized and rule-based weights with a scientifically valid anti-spoofing model, Module 4 was integrated with an open-weights pretrained **ASVspoof Light CNN (LCNN) / RawNet2 neural model** downloaded from HuggingFace (`caa-speech-detection-asvspoof2019/lcnn`). 

The system was evaluated against legitimate real audio presentation attack datasets—**ASVspoof 2017 (Physical Access Replay)** and **ASVspoof 2015 (Logical Access TTS and Voice Conversion)**—supplemented by **OpenSLR Mini-LibriSpeech** for bona fide human speech, using strict subject-disjoint Development/Calibration (DEV) and Held-Out Test (TEST) splits.

All automated system unit/integration tests (**35/35 passed**) and the production frontend build (**Vite build complete, 0 errors**) succeeded cleanly.

> [!IMPORTANT]
> **MANDATORY SCIENTIFIC VALIDATION STATEMENT:**  
> *"Software implementation verified; biometric generalization not yet scientifically established."*

---

### 1. Architectural Scope & Lock Integrity

In strict compliance with project directives:
- **Module 1 (DeepQShield):** **LOCKED & UNTOUCHED**
- **Module 2 (Face Identity):** **LOCKED & UNTOUCHED**
- **Module 3 (Face Liveness):** **LOCKED & UNTOUCHED**
- **Module 4 (Voice Authentication):** **UNLOCKED**
- **Module 5 (Multimodal Fusion):** **NOT STARTED**

---

### 2. Pretrained Audio Anti-Spoofing Model Architecture

#### 2.1 Model & Weight Provenance
- **Model Architecture:** Light Convolutional Neural Network (LCNN) with Max-Feature-Map (MFM) activation and 64-dimensional acoustic descriptor projection.
- **Pretrained Weights Source:** HuggingFace Hub (`caa-speech-detection-asvspoof2019/lcnn`, `best.pt`).
- **License:** Creative Commons Attribution 4.0 International / Open Data.
- **Pretrained Status:** Verified at runtime (`is_pretrained = True` asserted in `test_pad_classification`).

#### 2.2 Supplementary Physical Acoustic Safeguards
The pretrained neural model operates alongside supplementary physical domain acoustic guards:
- **Fundamental Frequency (F0) Pitch Stability:** Detects unnaturally flat pitch contours characteristic of legacy TTS algorithms ($f_{0, \text{mean}} > 50\text{ Hz}$ and $f_{0, \text{std}} < 1.5\text{ Hz}$).
- **High-Frequency Energy Ratio:** Identifies high-frequency acoustic artifacts ($>3.5\text{ kHz}$) typical of loudspeaker replay channels.
- **Zero-Crossing Rate (ZCR):** Detects high-frequency noise injection from acoustic playback hardware.

---

### 3. Real Audio PAD Dataset Provenance & Protocol

#### 3.1 Dataset Sources & Provenance
| Dataset Name | Provenance & Source | Attack / Audio Category | License |
| :--- | :--- | :--- | :--- |
| **OpenSLR Mini-LibriSpeech (`dev-clean-2`)** | OpenSLR Resource #31 | `BONAFIDE` Real Human Speech | CC BY 4.0 |
| **ASVspoof 2017 (`SpoofDetection_ASVspoof2017`)** | HuggingFace Hub / DynamicSuperb | `REPLAY_ATTACK` Physical Replay | CC BY 4.0 / Open Data |
| **ASVspoof 2015 (`SpoofDetection_ASVspoof2015`)** | HuggingFace Hub / DynamicSuperb | `SYNTHETIC_TTS` & `VOICE_CLONE` | CC BY 4.0 / Open Data |

#### 3.2 Subject-Disjoint Dataset Partitioning
All samples across the four audio categories were partitioned into strictly disjoint splits:
- **Development / Calibration Split (DEV - 40%):** Used exclusively for operating threshold selection ($\tau_{\text{liveness}}^* = 0.4900$).
- **Held-Out Test Split (TEST - 60%):** Evaluated strictly with zero post-hoc threshold tuning.

Sample counts per split:
- `BONAFIDE`: 11 DEV / 17 HELD-OUT TEST
- `REPLAY_ATTACK`: 73 DEV / 110 HELD-OUT TEST
- `SYNTHETIC_TTS`: 38 DEV / 58 HELD-OUT TEST
- `VOICE_CLONE`: 37 DEV / 56 HELD-OUT TEST

---

### 4. Empirical Held-Out Evaluation Metrics (ISO/IEC 30107-3 Standard)

Empirical evaluation results on the untouched Held-Out Test split:

| Metric | Result | Description / Target | Status |
| :--- | :--- | :--- | :--- |
| **Total Bona Fide Test Samples** | 17 | Real human speech recordings | Evaluated |
| **Total Attack Test Samples** | 224 | Physical replay, TTS & Voice Clone | Evaluated |
| **BPCER (Bona Fide Error Rate)** | **52.94%** (0.5294) | Genuine speech rejected as attack | High Noise Sensitivity |
| **Overall APCER (Attack Error Rate)** | **66.96%** (0.6696) | Presentation attacks accepted | Needs Domain Adaption |
| **Average ACER** | **59.95%** (0.5995) | $(APCER + BPCER) / 2$ | Baseline Uncalibrated |
| **PAD Equal Error Rate (EER)** | **59.32%** (0.5932) | Error rate where APCER = BPCER | Uncalibrated Channel |
| **PAD ROC-AUC** | **0.3146** | Area under ROC curve | Out-of-Domain Channel |
| **CPU Inference Latency** | **129.98 ms** | Average single-utterance CPU latency | Real-Time Capable |

#### Per-Attack Category APCER Breakdown
- **REPLAY_ATTACK (Physical Loudspeaker Replay):** APCER = **57.27%** (0.5727)
- **SYNTHETIC_TTS (Speech Synthesis / Text-to-Speech):** APCER = **74.14%** (0.7414)
- **VOICE_CLONE (Voice Conversion / Neural Clone):** APCER = **78.57%** (0.7857)

---

### 5. Dynamic Challenge & Security Verification Audit

- **Server-Generated Challenge:** Dynamic 3-word challenge phrase generated per session by backend (`VoiceSessionManager`).
- **Session Binding & Expiry:** Cryptographic session tokens with 300-second TTL. Re-use attempts return `SESSION_ALREADY_USED`.
- **Phrase Verification Status:** Retained as `phrase_match: "UNVALIDATED"`. Spoken phrase verification requires a dedicated Automatic Speech Recognition (ASR) model (e.g. OpenAI Whisper / Wav2Vec2), which was intentionally excluded from this pass per prompt directives.
- **Encrypted Template Storage:** Centroids are encrypted using **AEAD AES-256-GCM** prior to storage. Plaintext embeddings are never saved to disk.

---

### 6. Limitations & Claims Governance

1. **No Claim of Production-Grade Anti-Spoofing:** In accordance with audit directives, Module 4 does **not** claim "production-grade anti-spoofing" because held-out ACER on out-of-domain acoustic channels remains high (59.95%).
2. **ASR Integration Pending:** Phrase-matching challenge verification is unvalidated until an ASR transcription engine is integrated.

---

### 7. Final Lock Decision

**Module 4 MUST REMAIN UNLOCKED.**

The empirical PAD evaluation demonstrates that baseline open-weights models trained on synthetic benchmark conditions do not generalize out-of-the-box to arbitrary unseen acoustic channels without domain adaptation or fine-tuning. Therefore, Module 4 cannot be locked until fine-tuning and ASR verification are completed.

#### Verification Summary
- **Voice Unit & Integration Suite (`pytest tests/voice/test_voice_auth.py`):** **10/10 Passed (100%)**
- **Full System Test Suite (`pytest tests/`):** **35/35 Passed (100%)**
- **Frontend Production Build (`npm --prefix frontend run build`):** **Success (Vite build complete, 0 errors)**
- **Modules 1, 2, and 3:** **LOCKED & UNTOUCHED**
- **Module 5:** **NOT STARTED**

---
*Report compiled automatically by DeepGuard System Validation Suite.*
