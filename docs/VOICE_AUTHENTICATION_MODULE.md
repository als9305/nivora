# Module 4 — Voice Authentication & Audio Anti-Spoofing Technical Documentation

## 1. Objective
Module 4 establishes whether a presented voice sample originates from a **bona fide live human speaker** enrolled under a target identity, or from a **speaker mismatch**, **synthetic TTS / voice clone**, or **replayed audio presentation attack**. Module 4 operates as an independent security module within the Nivora platform.

Modules 1 (DeepQShield), 2 (Face Identity), and 3 (Face Liveness) remain **100% locked and untouched**.

## 2. Product Rule — Live Sensor Input Only
> **Strict Product Rule**: Nivora's production biometric authentication interface accepts biometric evidence only through live camera and microphone capture. User-selected biometric files are not accepted through the production authentication interface.
> - **Face Biometrics**: Camera / Webcam only
> - **Voice Biometrics**: Microphone only
> - **File Upload Prohibition**: Zero file-upload controls (`<input type="file">`, file pickers, or file drag-and-drop) exist in the production Voice Authentication user interface.

---

## 3. Threat Model
Voice Presentation Attack Detection (PAD) and Speaker Verification address adversaries attempting to bypass audio authentication sensors:
- **Zero-Touch Audio Attacks**: Adversaries playing pre-recorded human speech over secondary loudspeakers or mobile devices (Replay Attacks).
- **Synthetic Text-to-Speech (TTS) Attacks**: Neural or parametric speech synthesis generators producing robotic, flat-pitch trajectories ($F_0$ std dev $< 1.5\text{ Hz}$).
- **Voice Clone / Deepfake Audio Attacks**: Neural vocoder re-synthesized audio attempting to mimic a targeted speaker's timbre.

---

## 3. Architecture
The Module 4 processing pipeline consists of 4 distinct security layers:
```
Audio Payload (WAV / Base64)
       │
       ▼
[1] Audio Preprocessor & Quality Auditor (`models/voice_auth.py`)
       ├── Mono Downmix & Amplitude Normalization
       ├── Short-Time Energy VAD (Voice Activity Detection)
       └── Quality Audit (Duration >= 0.3s, RMS Energy, Clipping < 35%)
       │
       ▼
[2] Multi-Domain Acoustic Feature Extractor
       ├── 24-Band Log-Mel Spectrogram
       ├── 13-Bin MFCCs & Delta Coefficients
       ├── F0 Pitch Autocorrelation & Pitch Trajectory Std Dev
       ├── Spectral Centroid, Rolloff, Flux & Zero Crossing Rate (ZCR)
       └── High-Frequency Energy Ratio (>3.5 kHz)
       │
       ▼
[3] PyTorch Neural Models (`models/voice_auth.py`)
       ├── VoiceEmbeddingNet (128-d L2-Normalized Speaker Vector)
       └── PassiveAudioAntiSpoofNet (2-Class Softmax Classifier)
       │
       ▼
[4] Dual-Verdict Decision Engine
       ├── Speaker Verification (`speaker_verdict`: MATCH / MISMATCH)
       ├── Audio Liveness (`liveness_verdict`: LIVE / SPOOF / INSUFFICIENT_EVIDENCE)
       └── Final Authentication (`final_verdict`: AUTHENTICATED / REJECTED)
```

---

## 4. Preprocessing & Quality Audit
1. **Decoding**: Parses 8/16/24/32-bit int or float PCM WAV containers.
2. **Mono Downmix**: Averages multi-channel audio to single mono channel.
3. **Amplitude Normalization**: Scales signal to max amplitude $1.0$.
4. **Voice Activity Detection (VAD)**: Short-Time Energy frame windowing filters silence segments.
5. **Quality Audit**: Rejects recordings $< 0.30\text{s}$, RMS energy $< 0.01$, or clipping ratio $> 0.35$.

---

## 5. Speaker Enrollment & AEAD Template Protection
- **Multi-Sample Enrollment**: Accepts 3–5 independent speech recordings per user.
- **Centroid Aggregation**: Computes unit L2-normalized centroid embedding vector $\mathbf{c} = \frac{\frac{1}{N}\sum \mathbf{e}_i}{\|\frac{1}{N}\sum \mathbf{e}_i\|_2}$.
- **AEAD Encryption**: Templates are encrypted using **AES-256-GCM** via `VoiceTemplateRepository` before storage. Raw audio and plaintext embeddings are **never stored**.

---

## 6. Speaker Verification & Cosine Similarity
- Probe embedding $\mathbf{e}_{probe}$ is compared against decrypted template centroid $\mathbf{c}$:
  $$\text{Cosine Similarity}(\mathbf{e}_{probe}, \mathbf{c}) = \frac{\mathbf{e}_{probe} \cdot \mathbf{c}}{\|\mathbf{e}_{probe}\|_2 \|\mathbf{c}\|_2}$$
- Threshold: $\tau_{speaker}^* = 0.6500$.
- `speaker_verdict`: `MATCH` if similarity $\ge 0.65$, else `MISMATCH`.

---

## 7. Passive Audio Anti-Spoofing (PAD)
- **PyTorch Classifier**: `PassiveAudioAntiSpoofNet` processes 64-d acoustic feature vector.
- **Physical Acoustic Guards**:
  - `SYNTHETIC_TTS`: Flagged if $F_0$ mean $> 50\text{ Hz}$ and $F_0$ std $< 1.5\text{ Hz}$ (flat robotic pitch).
  - `REPLAY_ATTACK`: Flagged if high-frequency energy ratio $> 0.25$ or ZCR $> 0.22$.
- Threshold: $\tau_{liveness}^* = 0.5000$.

---

## 8. REST API Interface (`/api/v1/voice`)
- `POST /api/v1/voice/enroll`: Enrolls speaker template.
- `POST /api/v1/voice/verify`: Evaluates speaker verification & audio liveness.
- `GET /api/v1/voice/template/{user_id}`: Retrieves template status.
- `DELETE /api/v1/voice/template/{user_id}`: Deletes enrolled template.

---

## 9. Frontend Console (`/voice`)
- UI Component: [VoiceAuthConsole.jsx](file:///d:/projects/DeepGuard/frontend/src/components/VoiceAuthConsole.jsx)
- Page Route: `/voice` ([VoiceAuthPage.jsx](file:///d:/projects/DeepGuard/frontend/src/pages/VoiceAuthPage.jsx))
- Features: Microphone recording stream via WebRTC, WAV file upload fallback, real-time speaker match badge, liveness score, and acoustic spectral metrics.

---

## 10. Verification & Test Results
- **Module 4 Unit Test Suite**: **11/11 Passed (100%)** ([tests/voice/test_voice_auth.py](file:///d:/projects/DeepGuard/tests/voice/test_voice_auth.py)).
- **Full Project Test Suite**: **36/36 Passed (100%)** across Modules 1, 2, 3, and 4.
- **Frontend Production Build**: **Passed cleanly** (`built in 2.27s`).
- **Benchmark Evaluation Report**: Saved to [reports/voice_auth_benchmark.json](file:///d:/projects/DeepGuard/reports/voice_auth_benchmark.json).
