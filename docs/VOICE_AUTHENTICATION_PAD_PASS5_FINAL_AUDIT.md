# Voice Authentication & Audio Anti-Spoofing — Pass 5 Final Calibration & Generalization Audit

**MODULE 4 STATUS: UNLOCKED**

> [!IMPORTANT]
> **Module Boundaries & System Constraints:**
> - **Module 1 (DeepQShield):** 100% LOCKED, UNTOUCHED, AND NOT MODIFIED.
> - **Module 2 (Face Identity):** 100% LOCKED, UNTOUCHED, AND NOT MODIFIED.
> - **Module 3 (Face Liveness):** 100% LOCKED, UNTOUCHED, AND NOT MODIFIED.
> - **Module 5 (Multimodal Fusion):** NOT STARTED / UNTOUCHED.
> - **Module 4 Status:** **UNLOCKED** — while ROC-AUC (0.8472) and APCER (14.29% - 19.20%) meet the target thresholds, BPCER (29.41%) and ACER (21.85% - 24.30%) exceed the strict 20.0% objective lock criteria.
> - **Liveness Threshold Calibration:** Calibrated exclusively on DEV split ($\tau^*_{\text{DEV}} = 0.5300$). Evaluated on untouched HELD-OUT TEST split.
> - **Phrase Verification:** Remains `phrase_match = "UNVALIDATED"`. No ASR implemented.

---

## 1. Audit of the 5 False-Rejected Bona-Fide TEST Samples

At the baseline operating threshold ($\tau = 0.7000$), 5 out of 17 genuine TEST speech samples were rejected ($p_{\text{live}} < 0.7000$). Detailed inspection reveals the exact root cause:

| Sample ID | Dataset Source | $p_{\text{live}}$ Score | Duration | RMS Energy | $F_0$ std (Hz) | HF Energy Ratio | ZCR | Primary Root Cause / Override Mechanism |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Sample #2** | Mini-LibriSpeech | `0.1500` | 3.13s | 0.0353 | 75.48 | 0.0186 | `0.2286` | **Physical Guard Override:** ZCR > 0.22 forced $p_{\text{spoof}} = 0.85$ ($p_{\text{live}} = 0.15$). |
| **Sample #4** | Mini-LibriSpeech | `0.1500` | 3.44s | 0.0450 | 64.81 | 0.0237 | `0.2212` | **Physical Guard Override:** ZCR > 0.22 forced $p_{\text{spoof}} = 0.85$ ($p_{\text{live}} = 0.15$). |
| **Sample #11** | Mini-LibriSpeech | `0.1000` | 6.66s | 0.1484 | `0.00` | 0.0093 | 0.0831 | **Physical Guard Override:** $F_0$ std < 1.5 forced $p_{\text{spoof}} = 0.90$ ($p_{\text{live}} = 0.10$). |
| **Sample #12** | Mini-LibriSpeech | `0.3425` | 3.97s | 0.1243 | 113.97 | 0.0192 | 0.0855 | **Neural Net Under-Score:** Model assigned low liveness score based on acoustic feature vector. |
| **Sample #16** | Mini-LibriSpeech | `0.0045` | 5.89s | 0.1124 | `0.00` | 0.0020 | 0.0486 | **Physical Guard Override:** $F_0$ std < 1.5 forced $p_{\text{spoof}} = 0.90$ ($p_{\text{live}} = 0.10$). |

### Diagnostic Finding:
**80% of false rejections (4 out of 5 samples)** were caused by static physical acoustic guard overrides (`zcr > 0.22` and `f0_std < 1.5`) overriding neural network output on legitimate human speech segments (e.g. unvoiced fricatives or monotone speech where pitch estimation failed).

---

## 2. DEV-Only Threshold Calibration

Score distributions and optimal threshold search were executed **exclusively on the DEV split** (n=50 BONAFIDE, n=148 SPOOF):

### DEV Score Distributions
- **DEV Bona Fide:** $\text{Mean} = 0.4341$, $\text{Median} = 0.1740$, $\text{Std} = 0.4271$, $\text{Min} = 0.0000$, $\text{Max} = 0.9945$
- **DEV Spoof:** $\text{Mean} = 0.1827$, $\text{Median} = 0.0252$, $\text{Std} = 0.2903$, $\text{Min} = 0.0000$, $\text{Max} = 0.9929$

### Calibrated DEV Threshold Operating Points
- **Min-ACER Optimal Threshold ($\tau^*_{\text{DEV}}$):** `0.5300` (DEV ACER = 34.43%)
- **Threshold for APCER $\le 10\%$:** `0.7300`
- **Threshold for APCER $\le 15\%$:** `0.5300`
- **Threshold for APCER $\le 20\%$:** `0.4350`

---

## 3. Evaluation on Untouched HELD-OUT TEST Set (N = 241)

Evaluating the untouched HELD-OUT TEST set (17 BONAFIDE, 224 SPOOF) across candidate DEV-calibrated operating points:

| Metric | DEV Min-ACER ($\tau = 0.5300$) | Pass 4 Baseline ($\tau = 0.7000$) | DEV APCER $\le 20\%$ ($\tau = 0.4350$) |
| :--- | :---: | :---: | :---: |
| **BPCER (Bona-Fide Error)** | **29.41%** (5/17) | **29.41%** (5/17) | **29.41%** (5/17) |
| **Overall APCER (Attack Error)** | **19.20%** (43/224) | **14.29%** (32/224) | **21.88%** (49/224) |
| **Average ACER** | **24.30%** | **21.85%** | **25.64%** |
| **ROC-AUC** | **0.8472** | **0.8472** | **0.8472** |
| **Equal Error Rate (EER)** | **24.04%** | **24.04%** | **24.04%** |
| **Replay APCER** | 10.91% (12/110) | 9.09% (10/110) | 12.73% (14/110) |
| **TTS APCER** | 31.03% (18/58) | 27.59% (16/58) | 32.76% (19/58) |
| **Voice Clone APCER** | 23.21% (13/56) | 10.71% (6/56) | 28.57% (16/56) |

### Confusion Matrix ($\tau = 0.5300$)
```
                       PREDICTED
                  LIVE          SPOOF
ACTUAL  LIVE      12            5        (BPCER = 29.41%)
        SPOOF     43            181      (APCER = 19.20%)
```

---

## 4. Objective Lock Decision Check

Evaluating against the required objective locking criteria on TEST:

| Criterion | Target Requirement | Measured Value (DEV $\tau^* = 0.5300$) | Evaluation Status |
| :--- | :---: | :---: | :---: |
| **1. ROC-AUC** | $\ge 0.80$ | `0.8472` | **PASS** |
| **2. ACER** | $\le 20.0\%$ | `24.30%` | **FAIL** |
| **3. BPCER** | $\le 20.0\%$ | `29.41%` | **FAIL** |
| **4. APCER** | $\le 20.0\%$ | `19.20%` | **PASS** |

### Decision Summary
Because Criteria 2 (ACER = 24.30% > 20.0%) and Criteria 3 (BPCER = 29.41% > 20.0%) do not satisfy the strict lock rules:

**MODULE 4 STATUS: UNLOCKED**

---

## 5. Primary Bottleneck & Highest-Value Recommendation

### Remaining Bottleneck
The hardcoded physical acoustic guard heuristics in `evaluate_anti_spoof` (`f0_std < 1.5` -> $p_{\text{spoof}}=0.90$ and `zcr > 0.22` -> $p_{\text{spoof}}=0.85$) cause **80% of false rejections** on genuine human speech. These hard bounds penalize unvoiced speech sounds (fricatives/plosives) and steady vocalizations.

### Recommended Next Correction (Single Highest Value)
**Soften and refine the physical acoustic guard rules** by converting binary hard overrides into continuous penalty scaling (or removing `zcr > 0.22` / `f0_std < 1.5` hard overrides in favor of pure neural score prediction with learned acoustic features).

---

## 6. Security Requirements Verification

- **Microphone-Only Input:** Verified in `RecordControls.tsx` (`navigator.mediaDevices.getUserMedia`).
- **No File Upload / Drag-and-Drop:** Verified zero upload UI or endpoint accepting uploaded audio files.
- **Audio Transport:** 16 kHz mono PCM WAV over HTTP POST API (`/api/voice/verify`).
- **Server Challenge Verification:** 300-second TTL single-use challenge token validated before processing.
- **Data Privacy & Storage:** Raw audio bytes processed in-memory and discarded. No raw audio stored to disk.
- **Template Security:** AEAD AES-256-GCM encrypted speaker embedding templates.
- **Phrase Verification:** Confirmed `phrase_match = "UNVALIDATED"`.

---

## 7. Verification Test Suite Status

```powershell
# Voice Auth Module Unit Tests
python -m pytest tests/voice/test_voice_auth.py -v
# Output: 10 passed in 6.91s

# Full Repository Test Suite
python -m pytest tests/ -v
# Output: 35 passed in 8.59s (api, face/identity, face/liveness, voice/voice_auth)

# Frontend Production Build
cmd /c npm --prefix frontend run build
# Output: dist/ assets generated cleanly in 2.30s
```

---
*Report generated automatically following scientific audit standards.*
