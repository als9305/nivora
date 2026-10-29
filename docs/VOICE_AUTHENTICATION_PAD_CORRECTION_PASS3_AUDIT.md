# Module 4 — Voice Authentication & Audio Anti-Spoofing
## Audio Presentation Attack Detection (PAD) Correction Pass 3 Audit Report

**System Name:** Nivora DeepGuard Security Platform  
**Module Under Audit:** Module 4 — Voice Authentication & Audio Anti-Spoofing  
**Audit Type:** PAD Correction Pass 3 — Targeted BPCER Balance & DEV Distribution Audit  
**Date:** September 21, 2026  
**Module Status:** **UNLOCKED** (Software Implementation Verified; Biometric Generalization Pending)  

---

### Executive Summary

This report documents **Correction Pass 3** for **Module 4 (Voice Authentication & Audio Anti-Spoofing)**.

The objective of Correction Pass 3 was to address the high Bona Fide Presentation Classification Error Rate (BPCER = 82.35%) observed in Pass 2 without destroying presentation attack detection capability (where APCER reached 9.82%).

Step 1 & 2 conducted an audit of training labels, class weights, and DEV score distributions. The audit revealed that the extreme class weighting ratio ($w_0 / w_1 = 13.45$) used in Pass 1/v1 severely distorted logit optimization, causing threshold calibration to select $\tau^* = 0.6300$, which bisected the bona-fide score distribution.

Step 5 introduced a balanced training objective in `scripts/train_pad_model_v2.py` utilizing moderate class weighting ($w_0 = 3.0, w_1 = 1.0$) and Live batch oversampling. The new checkpoint was saved to `models/trained_pad_model_v2.pt`.

All automated system tests (**35/35 passed**) and the production frontend build (**Vite build complete, 0 errors**) succeeded cleanly.

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

### 2. Section 1 — Audit of Training Labels & Class Weights

#### 2.1 Training Label Verification
Target label assignments in `scripts/train_pad_model_v2.py` were verified as follows:
- **Class 0 (`BONAFIDE`):** Real human speech recordings (Target = 0).
- **Class 1 (`SPOOF`):** Presentation attacks (Replay, TTS, Voice Clone; Target = 1).

#### 2.2 Class Weight Audit & Calculation
- **DEV Split Sample Counts:** 11 Bona Fide samples ($N_{\text{live}}$), 148 Spoof samples ($N_{\text{spoof}}$), Total = 159 samples.
- **Pass 2 (v1) Weight Calculation:**
  $$w_{\text{live}} = \frac{159}{2 \times 11} = 7.2273, \quad w_{\text{spoof}} = \frac{159}{2 \times 148} = 0.5372, \quad \text{Ratio } \frac{w_{\text{live}}}{w_{\text{spoof}}} = 13.45$$
- **Diagnostic Finding:** The $13.45\times$ weight penalty forced the optimizer into extreme logit dynamic ranges, driving the DEV-calibrated threshold to $\tau^* = 0.6300$, which cut directly through the middle of the genuine speech distribution.
- **Pass 3 (v2) Corrected Objective:** Replaced extreme weighting with moderate class weighting ($w_{\text{live}} = 3.0, w_{\text{spoof}} = 1.0$) combined with 4x Live batch oversampling (192 batch size per epoch).

---

### 3. Section 2 — Empirical DEV Split Score Distribution Audit

The score statistical breakdown ($p_{\text{live}}$) on the Development/Calibration (DEV) split:

| Class Category | Sample Count ($n$) | Min | Max | Mean | Median | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BONAFIDE (Live)** | 11 | 0.1500 | 0.8056 | **0.6768** | **0.7219** | 0.1768 |
| **SPOOF (Attacks)** | 148 | 0.0041 | 0.8226 | **0.1744** | **0.0159** | 0.2647 |

- **Distribution Overlap:** Score range overlap on DEV is $[0.1500, 0.8056]$.
- **Threshold Selection:** Calibrated operating threshold ($\tau_{\text{liveness}}^*$) was evaluated exclusively on DEV to minimize DEV ACER.

---

### 4. Section 3 — Ingestion Pipeline & Security Preservation

The core production authentication pipeline was strictly preserved:
- **Microphone-Only Ingestion:** Live device stream only (`MicCapture.jsx`, `VoiceAuthConsole.jsx`); zero production file-upload paths.
- **16 kHz PCM Audio Specs:** 16 kHz 16-bit mono RIFF WAV transport with native `soundfile` fallback.
- **Session Security:** Dynamic 3-word challenge phrase, 300s TTL, single-use anti-replay token (`SESSION_ALREADY_USED`).
- **Cryptographic Protection:** AEAD AES-256-GCM encryption for voice centroids.
- **Challenge Verification:** Retained as `phrase_match = "UNVALIDATED"`.

---

### 5. Section 4 — Empirical Held-Out Performance Metrics Comparison (v1 vs v2)

Evaluation results on the untouched Held-Out TEST split (241 samples: 17 Bona Fide, 224 Attacks):

| Metric | Pass 2 Baseline (v1) | Pass 3 Corrected (v2) | Comparison & Analysis |
| :--- | :--- | :--- | :--- |
| **DEV Calibrated Threshold ($\tau^*$)** | 0.6300 | **0.6700** | Calibrated exclusively on DEV |
| **Total Bona Fide TEST Samples** | 17 | **17** | Equal Split |
| **Total Attack TEST Samples** | 224 | **224** | Equal Split |
| **Overall APCER (Attack Error)** | **9.82%** | **47.77%** | Tradeoff vs BPCER |
| **BPCER (Bona Fide Error Rate)** | 82.35% | **29.41%** | **-52.94% Absolute BPCER Reduction** |
| **Average ACER** | 46.09% | **38.59%** | **-7.50% Absolute ACER Improvement** |
| **Replay APCER (`REPLAY_ATTACK`)** | **9.09%** | **27.27%** | Physical Replay Defense |
| **Synthetic TTS APCER (`SYNTHETIC_TTS`)** | **10.34%** | **72.41%** | TTS Defense |
| **Voice Clone APCER (`VOICE_CLONE`)** | **10.71%** | **62.50%** | Neural Voice Clone Defense |
| **PAD EER** | 63.38% | **41.12%** | **-22.26% EER Improvement** |
| **PAD ROC-AUC** | 0.4466 | **0.5814** | **+0.1348 ROC-AUC Gain** |
| **CPU Inference Latency** | 368.62 ms | **401.39 ms** | Real-Time Capable |

---

### 6. Section 5 — Final Lock Decision & Recommendations

**Module 4 MUST REMAIN UNLOCKED.**

While Pass 3 successfully demonstrated that balanced class weighting reduces BPCER from 82.35% to 29.41% (improving overall ROC-AUC to 0.5814 and ACER to 38.59%), the trade-off between APCER and BPCER confirms that out-of-domain acoustic channel variance requires multi-channel training data and ASR phrase verification.

#### Verification Summary
- **Voice Unit & Integration Suite (`pytest tests/voice/test_voice_auth.py`):** **10/10 Passed (100%)**
- **Full System Test Suite (`pytest tests/`):** **35/35 Passed (100%)**
- **Frontend Production Build (`npm --prefix frontend run build`):** **Success (Vite build complete, 0 errors)**
- **Modules 1, 2, and 3:** **LOCKED & UNTOUCHED**
- **Module 5:** **NOT STARTED**

---
*Report compiled automatically by DeepGuard System Validation Suite.*
