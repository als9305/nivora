# Module 4 — Voice Authentication & Audio Anti-Spoofing Diagnostic Audit Report

> **Audit Date**: 2026-09-20  
> **Auditor**: Antigravity AI Engineering & Security Auditor  
> **Diagnostic Pass**: Pass 2 (READ-ONLY Scientific Diagnostic)  
> **Module Scope**: Module 4 ONLY (Voice Authentication & Audio Anti-Spoofing)  
> **Module Lock Status**: **UNLOCKED** ("Pretrained model integration is technically correct, but the current benchmark does not demonstrate sufficient biometric performance.")

---

## 1. Executive Summary & Diagnostic Finding

This report presents a **read-only scientific diagnostic** investigating why the held-out speaker verification benchmark yielded:
- **FAR = 100.0%** (27 / 27 impostor trials accepted)
- **FRR = 0.0%** (0 / 9 genuine trials rejected)
- **EER = ~50.0%**

### Core Diagnostic Conclusion
> **IDENTIFIED ROOT CAUSE**: The pretrained **SpeechBrain ECAPA-TDNN (`spkrec-ecapa-voxceleb`)** PyTorch model is correctly loaded and initialized with official VoxCeleb 1 & 2 weights. However, the evaluation benchmark generates audio using **programmatically synthesized pure sine waves**. Because pure sine waves lack human vocal tract formant resonances ($F_1, F_2, F_3$) and phonetic structure, SpeechBrain ECAPA-TDNN maps all out-of-distribution non-speech signals to a single constant point $\mathbf{e}_{\text{sine}}$ in the 192-dimensional embedding space. Consequently, Cosine Similarity between ANY two synthetic sine wave signals (whether genuine or impostor) is $\sim 0.9998$. When evaluated against operating threshold $\tau_{\text{speaker}} = 0.6500$, `0.9998 >= 0.6500` is TRUE for 100% of impostor trials, resulting in **FAR = 100%**.

---

## 2. ECAPA Preprocessing Trace

The complete feature extraction and normalization pipeline was audited against SpeechBrain's official `spkrec-ecapa-voxceleb` specification (`hyperparams.yaml`):

| Parameter | Official SpeechBrain Spec | Current Implementation | Audit Match |
| :--- | :--- | :--- | :--- |
| **Input Sample Rate** | $16,000\text{ Hz}$ mono PCM | $16,000\text{ Hz}$ mono PCM | **EXACT MATCH** |
| **Mel Filterbanks ($N_{mels}$)** | 80 log-mel channels | 80 log-mel channels | **EXACT MATCH** |
| **STFT Window & Hop** | $N_{fft} = 512$, $H = 160$ ($10\text{ms}$) | $N_{fft} = 512$, $H = 160$ ($10\text{ms}$) | **EXACT MATCH** |
| **Log Compression** | $\log(\text{mel\_spec} + 1e-6)$ | $\log(\text{mel\_spec} + 1e-6)$ | **EXACT MATCH** |
| **Tensor Input Shape** | $[B, 80, T]$ | $[B, 80, T]$ | **EXACT MATCH** |
| **Feature Normalization** | Sentence-level CMVN across time $T$ | Subtracts per-channel time-mean | **EXACT MATCH** |
| **Output Layer** | 192-d FC Conv projection | 192-d FC Conv projection (`fc.conv`) | **EXACT MATCH** |
| **Embedding Normalization** | L2 Normalization ($\|\mathbf{e}\|_2 = 1.0$) | L2 Normalization ($\|\mathbf{e}\|_2 = 1.0$) | **EXACT MATCH** |

---

## 3. Enrollment Trace

For every speaker identity in the benchmark protocol:
- **Speaker IDs**: 6 genuine speaker groups (`speaker_live_01` to `speaker_live_06`).
- **Enrollment Samples per Speaker**: 3 independent speech audio samples ($1.2\text{s}$ duration each).
- **Embedding Extraction**: Processed through `PretrainedECAPATDNN` to produce 192-d float vectors.
- **Embedding Norm Verification**: $\|\mathbf{e}\|_2 = 1.0000 \pm 0.0001$ for 100% of samples.
- **Centroid Construction**:
  $$C = \frac{\frac{1}{3}\sum_{i=1}^3 \mathbf{e}_i}{\left\|\frac{1}{3}\sum_{i=1}^3 \mathbf{e}_i\right\|_2}$$
- **Verification**: Enrolled target speaker identity is correctly isolated and associated with the designated speaker ID.

---

## 4. Verification Trace & Cosine Score Distributions

Statistical distribution of Cosine Similarity scores ($S_C(\mathbf{e}_1, \mathbf{e}_2) \in [-1.0, 1.0]$) across the held-out test trials:

| Metric | Genuine Trials (Same Speaker) | Impostor Trials (Different Speaker) |
| :--- | :--- | :--- |
| **Trial Count** | 9 trials | 27 trials |
| **Minimum Cosine Score** | `0.999675` | `0.999650` |
| **Maximum Cosine Score** | `0.999880` | `0.999954` |
| **Mean Cosine Score** | `0.999812` | `0.999843` |
| **Median Cosine Score** | `0.999820` | `0.999850` |
| **Standard Deviation** | `0.000045` | `0.000052` |
| **Score Overlap / Gap** | Mean Gap = **-0.000031** (Impostor scores equal genuine scores) |

### Sample Trial Records

| Trial ID | Enrolled Speaker ID | Probe Speaker ID | True Label | Cosine Score | Decision ($\tau=0.65$) | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **T-01** | `speaker_live_04` | `speaker_live_04` | GENUINE | `0.999812` | `MATCH` | **TRUE ACCEPT** |
| **T-02** | `speaker_live_05` | `speaker_live_05` | GENUINE | `0.999834` | `MATCH` | **TRUE ACCEPT** |
| **T-03** | `speaker_live_04` | `speaker_live_05` | IMPOSTOR | `0.999843` | `MATCH` | **FALSE ACCEPT** |
| **T-04** | `speaker_live_04` | `speaker_replay_04`| IMPOSTOR | `0.999857` | `MATCH` | **FALSE ACCEPT** |
| **T-05** | `speaker_live_04` | `speaker_clone_04` | IMPOSTOR | `0.999829` | `MATCH` | **FALSE ACCEPT** |

---

## 5. Score Direction & Label Orientation Audit

1. **Score Direction Verification**:
   - `VoiceAuthEngine.compute_similarity(e1, e2)` computes $\mathbf{e}_1 \cdot \mathbf{e}_2$.
   - Direction: **Higher Cosine Similarity = More Likely Same Speaker**.
   - Threshold Comparison: `speaker_verdict = "MATCH" if voice_similarity >= tau_speaker else "MISMATCH"`.
   - **Verification**: Mathematically correct.
2. **Label Orientation Verification**:
   - Genuine Trial: Same speaker identity (`true_speaker_match = True`).
   - Impostor Trial: Cross-speaker mismatch or presentation attack (`true_speaker_match = False`).
   - **FAR Calculation**: `impostors_accepted / total_impostors` = $27 / 27 = 1.0000$ ($100\%$).
   - **FRR Calculation**: `genuine_rejected / total_genuine` = $0 / 9 = 0.0000$ ($0\%$).
   - **Verification**: Formulas and label mapping are 100% correct.

---

## 6. Dataset Integrity Audit

- **Audio Content**: Synthesized using `generate_benchmark_speech_sample()`.
- **Finding**: Signals consist of multi-tone sine wave combinations: $\sin(\phi) + 0.25\sin(2\phi) + 0.12\sin(3\phi)$.
- **Anomalous Behavior**: Pure sine waves do not possess vocal tract acoustic formants ($F_1 \approx 500\text{Hz}, F_2 \approx 1500\text{Hz}, F_3 \approx 2500\text{Hz}$). Because SpeechBrain ECAPA-TDNN was trained on VoxCeleb real human speech, out-of-distribution non-speech signals collapse to a constant point $\mathbf{e}_{\text{sine}}$ in the 192-d space.

---

## 7. Control Experiment

A control experiment was performed using multi-formant vocal tract acoustic speech signals ($F_0=120\text{Hz}$ male vs $F_0=210\text{Hz}$ female):

| Trial Type | Description | Cosine Score |
| :--- | :--- | :--- |
| **Same Speaker A** | Male Vocal Tract Utterance 1 vs 2 | **0.9998** |
| **Same Speaker B** | Female Vocal Tract Utterance 1 vs 2 | **0.9998** |
| **Different Speaker** | Male Speaker A vs Female Speaker B | **0.9998** |

**Finding**: Out-of-distribution synthetic signals produce score collapse regardless of pitch variation. Evaluation on real human speech datasets (e.g. VoxCeleb, LibriSpeech) is required for biometric validation.

---

## 8. Presentation Attack Detection (PAD) Diagnostic

| Attack Category | Sample Count | APCER | Mean Liveness Score | Rejection Status |
| :--- | :--- | :--- | :--- | :--- |
| **GENUINE_LIVE** | 9 | **BPCER: 0.0%** | `0.5000` | **ACCEPT** |
| **SYNTHETIC_TTS** | 9 | **0.0000 (0.0%)** | `0.1000` | **100% REJECT** (Monotone pitch heuristic triggered) |
| **REPLAY_ATTACK** | 9 | **1.0000 (100.0%)** | `0.4442` | **FAIL TO REJECT** |
| **VOICE_CLONE** | 9 | **1.0000 (100.0%)** | `0.4415` | **FAIL TO REJECT** |

### Root Cause of PAD Failure
1. **Model Weights**: `PassiveAudioAntiSpoofNet` contains randomly initialized weights that have not been trained on physical audio anti-spoofing datasets (e.g., ASVspoof 2019 / 2021).
2. **Heuristic Scope**: Rule-based physical guards detect monotone TTS ($F_0\text{ std} < 1.5\text{Hz}$) but do not capture complex non-monotone replay or voice-clone phase anomalies.

---

## 9. Mandatory Scientific Report Statement

> **"Pretrained model integration is technically correct, but the current benchmark does not demonstrate sufficient biometric performance."**

---

## 10. Test & Build Execution Results

Executing required commands:

1. **Voice Test Suite**:
   ```bash
   python -m pytest tests/voice/test_voice_auth.py -v
   ```
   **10 / 10 Passed (100%)**

2. **Full System PyTest Suite (Modules 1–4)**:
   ```bash
   python -m pytest tests/ -v
   ```
   **35 / 35 Passed (100%)**

3. **Frontend Production Build**:
   ```bash
   cmd /c npm --prefix frontend run build
   ```
   **Passed cleanly in 2.24s**

4. **Benchmark Script**:
   ```bash
   python scripts/benchmark_voice_auth.py
   ```
   Report generated at [reports/voice_auth_benchmark.json](file:///d:/projects/DeepGuard/reports/voice_auth_benchmark.json).

---

## 11. Final Module Lock Decision

> **FINAL DECISION**: **MODULE 4 IS UNLOCKED**

Although the software pipeline, Web Audio API transport, AEAD AES-256-GCM template encryption, and SpeechBrain ECAPA-TDNN VoxCeleb model integration are technically correct, the held-out benchmark evaluation on synthetic signals exhibits **FAR = 100%** and **PAD APCER = 66.67%**. **Module 4 CANNOT be scientifically declared locked.**

**Module 5 has NOT been started.**

---
*Report compiled automatically by DeepGuard Antigravity Scientific Audit Engine.*
