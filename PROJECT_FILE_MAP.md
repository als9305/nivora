# NIVORA — Project File and Component Map

This document provides a comprehensive structural mapping of the **NIVORA** (DeepGuard) codebase at `D:\projects\DeepGuard`. Every core module, service, script, and component is categorized with its exact file path, primary responsibilities, key classes and functions, and system dependencies.

---

## 1. Directory Structure Overview

```
D:\projects\DeepGuard\
├── backend/                  # FastAPI Application Core
│   ├── app/
│   │   ├── core/             # Configuration, logging, security constants
│   │   ├── database/         # MongoDB client, repositories, in-memory fallbacks
│   │   ├── models/           # Pydantic schemas, domain entities
│   │   ├── routes/           # Core API route handlers (auth, verification)
│   │   ├── services/         # Orchestration & business logic services
│   │   └── main.py           # FastAPI server entry point
│   ├── run.py                # Backend server runner (uvicorn launcher)
│   └── tests/                # Backend unit and integration tests
├── api/                      # Modular API Routers (v1 endpoints)
│   ├── routes/
│   │   ├── face.py           # /api/v1/face
│   │   ├── liveness.py       # /api/v1/liveness
│   │   ├── deepfake.py       # /api/v1/deepfake
│   │   ├── voice.py          # /api/v1/voice
│   │   └── multimodal.py     # /api/v1/authenticate
│   └── dependencies.py       # FastAPI dependency injection singletons
├── models/                   # Neural Network Architectures & ML Engines
│   ├── attention.py          # Spatial & Channel Feature Attention
│   ├── classifier.py         # Multi-layer Perceptron Classification Head
│   ├── deepqshield.py        # DeepQShield End-to-End Deepfake Detector
│   ├── ecapa_voxceleb.ckpt   # ECAPA-TDNN pre-trained speaker weights (83.3 MB)
│   ├── face_liveness.py      # PassiveAntiSpoofNet & MultiDomainFeatureExtractor
│   ├── face_recognition.py   # InceptionResnetV1 Face Feature Extractor
│   ├── feature_fusion.py     # Gated Multimodal/Multiscale Feature Fusion
│   ├── lwe_layer.py          # Learning With Errors (LWE) Lattice Layer
│   ├── resnext_backbone.py   # ResNeXt50_32x4d Feature Extraction Backbone
│   ├── temporal_aggregator.py# Temporal Rolling Liveness & Deepfake Aggregator
│   ├── trained_pad_model_v4.pt # Audio PAD neural weights (74 KB)
│   └── voice_auth.py         # VoiceAuthEngine & PassiveAudioAntiSpoofNet
├── security/                 # Post-Quantum Cryptography & Biometric Crypto
│   ├── pqc/
│   │   ├── kyber.py          # Kyber-1024 Key Encapsulation Mechanism
│   │   ├── dilithium.py      # Dilithium-5 Digital Signature Scheme
│   │   └── provider.py       # PQCProvider high-level audit signing interface
│   ├── biometric_crypto.py   # AES-256-GCM AEAD Template Encryption Engine
│   └── hashing.py            # SHA3-256 and Canonical JSON serialization
├── video/                    # Video Processing & Forensic Extraction
│   ├── face_detector.py      # OpenCV Haar Cascade & MTCNN Face Detectors
│   ├── frame_extractor.py    # OpenCV Video Frame Decoders & Sampling
│   └── visualizer.py         # Forensic Bounding Box & Annotation Overlay
├── frontend/                 # Vite + React User Interface
│   ├── src/
│   │   ├── components/       # UI Widgets (Gauges, Consoles, Webcams)
│   │   ├── context/          # AuthContext & Session State Management
│   │   ├── pages/            # View Pages (Dashboard, Login, Enrolment, History)
│   │   ├── services/         # Axios API Client & Route Mappings
│   │   ├── App.jsx           # Master Layout, Header, Router
│   │   └── main.jsx          # React DOM entry point
│   ├── package.json          # Node.js dependencies & scripts
│   └── vite.config.js        # Vite bundler configuration
├── checkpoints/              # Trained PyTorch Neural Weights
│   ├── deepqshield_best.pth  # DeepQShield model weights (418 MB)
│   └── liveness_best.pth     # Face Liveness model weights (5.14 MB)
└── tests/                    # Project-Wide Comprehensive Test Suites
    ├── face/                 # Module 1 & 2 tests
    ├── voice/                # Module 3 voice recognition & PAD tests
    ├── deepfake/             # Module 4 DeepQShield verification tests
    ├── multimodal/           # Module 5 fusion & hard security gate tests
    └── security/             # PQC & AEAD encryption verification tests
```

---

## 2. Neural Network Models & Machine Learning Engines (`models/`)

| File Path | Primary Class / Function | Responsibility | Input Shape / Type | Output Shape / Type | Checkpoints / Dependencies |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `models/face_recognition.py` | `FaceRecognitionEngine` | Extracts 512-D identity embeddings from aligned facial crops; computes cosine similarity. | $(B, 3, 160, 160)$ normalized image tensor | $(B, 512)$ unit-normalized vector ($L_2$) | `facenet_pytorch.InceptionResnetV1(pretrained='vggface2')` |
| `models/face_liveness.py` | `PassiveAntiSpoofNet`<br>`MultiDomainFeatureExtractor`<br>`TemporalLivenessAggregator` | Evaluates facial presentation attack cues (Moiré, texture, color distribution, temporal micro-motion). | Tensor $(B, 3, 224, 224)$ + OpenCV frame sequence | Liveness score $[0.0, 1.0]$, Spoof probability, Verdict | `checkpoints/liveness_best.pth` (5.14 MB), OpenCV, NumPy |
| `models/voice_auth.py` | `VoiceAuthEngine`<br>`PassiveAudioAntiSpoofNet` | Extracts 192-D speaker embeddings and 64-D acoustic presentation defense features. | 16 kHz mono PCM float audio array / buffer | 192-D speaker vector, 64-D PAD acoustic descriptor, Spoof score | `models/ecapa_voxceleb.ckpt` (83.3 MB), `models/trained_pad_model_v4.pt` (74 KB), Librosa, PyTorch |
| `models/deepqshield.py` | `DeepQShieldModel` | Unified deepfake detection architecture fusing spatial representations with lattice LWE transformations. | $(B, 3, 224, 224)$ facial crop tensor | Classification logits $(B, 2)$, fake probability $[0.0, 1.0]$ | `checkpoints/deepqshield_best.pth` (418 MB) |
| `models/resnext_backbone.py` | `ResNeXtBackbone` | Feature extraction backbone based on ResNeXt-50 ($32\times4\text{d}$) architecture. | $(B, 3, H, W)$ image tensor | 2048-D latent feature tensor $(B, 2048)$ | `torchvision.models.resnext50_32x4d` |
| `models/lwe_layer.py` | `LWELayer` | Two-stage Learning With Errors projection layer simulating lattice encryption perturbations. | $(B, 2048)$ feature vector | 256-D perturbed lattice representation $(B, 256)$ | PyTorch, `torch.randn_like` (noise $\sigma=0.1$) |
| `models/feature_fusion.py` | `FeatureFusionModule` | Fuses raw CNN backbone features with transformed lattice LWE representations. | Backbone $(B, 2048)$ + LWE $(B, 256)$ | $(B, 2048)$ fused representation with residual gate | PyTorch Linear, BatchNorm1d, ReLU |
| `models/attention.py` | `FeatureAttention` | Spatial and channel-wise squeeze-and-excitation attention re-weighting salient facial regions. | $(B, 2048)$ fused representation | $(B, 2048)$ channel-attenuated representation | PyTorch Linear, Sigmoid, ReLU |
| `models/classifier.py` | `ClassifierHead` | Final classification MLP head with dropout regularization. | $(B, 2048)$ attended feature vector | $(B, 2)$ class logits (`[authentic, deepfake]`) | PyTorch Linear, Dropout ($p=0.5$), ReLU |
| `models/temporal_aggregator.py` | `TemporalAggregator` | Rolling sliding-window aggregator over video frame sequences for stable forensic verdicts. | Sequential sequence of frame anomaly scores | Mean anomaly score, peak variance, aggregate verdict | NumPy, collections.deque |

---

## 3. Post-Quantum & Classical Security Layer (`security/`)

| File Path | Key Classes / Functions | Responsibility | Standards / Primitives Used |
| :--- | :--- | :--- | :--- |
| `security/biometric_crypto.py` | `BiometricCryptoEngine` | Encrypts and decrypts face (512-D) and voice (192-D) floating-point embedding templates at rest. | AES-256-GCM AEAD, 12-byte random IV, 16-byte authentication tag, associated data verification. |
| `security/hashing.py` | `canonical_json`<br>`sha3_256_hash` | Generates deterministic byte representations and cryptographic digests for tamper-evident logs. | SHA3-256, NIST FIPS 202, canonical recursive key-sorted JSON serialization. |
| `security/pqc/kyber.py` | `Kyber1024` | Post-quantum Key Encapsulation Mechanism (KEM) simulation adhering to NIST ML-KEM byte lengths. | Public Key: 1568 bytes, Secret Key: 3168 bytes, Ciphertext: 1568 bytes, Shared Secret: 32 bytes. |
| `security/pqc/dilithium.py` | `Dilithium5` | Post-quantum Digital Signature Algorithm (DSA) simulation adhering to NIST ML-DSA-87 byte lengths. | Public Key: 2592 bytes, Secret Key: 4896 bytes, Signature: 4595 bytes. Deterministic SHA3-256 hashing. |
| `security/pqc/provider.py` | `PQCProvider` | High-level orchestrator generating signed and encapsulated forensic audit records for every verification event. | Integrates Dilithium-5 digital signatures and Kyber-1024 key encapsulation over canonical SHA3 digests. |

---

## 4. Backend Application Core & Orchestration (`backend/app/`)

| File Path | Key Components | Responsibility |
| :--- | :--- | :--- |
| `backend/app/main.py` | `FastAPI` app initialization | Root application bootstrap; registers CORS middleware, mounts `/api/auth`, `/api/verification`, and `/api/v1/*` routes, manages lifecycle events. |
| `backend/app/core/config.py` | `Settings` (Pydantic BaseSettings) | Global configuration: MongoDB URI, database name, JWT secret, PQC enablement, biometric encryption key, default security thresholds. |
| `backend/app/database/mongodb.py` | `get_database`<br>`connect_to_mongo` | Async MongoDB client via `motor.motor_asyncio`. Implements seamless fallback to in-memory dictionary storage if MongoDB is offline. |
| `backend/app/database/repositories.py` | `UserRepository`<br>`FaceTemplateRepository`<br>`VoiceTemplateRepository` | Data persistence abstractions for user accounts, AES-GCM encrypted biometric templates, and forensic audit logs. |
| `backend/app/services/multimodal_fusion_service.py` | `MultimodalFusionService` | **Module 5 Final Verification Engine**: Evaluates evidence from Modules 1–4, enforces Hard Security Gates, computes dynamic risk score and assurance level. |
| `backend/app/services/face_service.py` | `FaceService` | Orchestrates Module 1 Face Identity (feature extraction, template comparison, template enrollment with AES-256-GCM encryption). |
| `backend/app/services/liveness_service.py` | `LivenessService` | Orchestrates Module 2 Face Liveness (extracts spatial and frequency anti-spoofing cues, runs temporal micro-motion evaluation). |
| `backend/app/services/deepfake_service.py` | `DeepfakeService` | Orchestrates Module 4 Deepfake Detection (executes DeepQShieldModel inference, attaches PQC signature audit records). |
| `backend/app/services/voice_service.py` | `VoiceService` | Orchestrates Module 3 Voice Identity (ECAPA-TDNN speaker verification, PassiveAudioAntiSpoofNet PAD scoring, challenge phrase ASR checks). |
| `backend/app/services/speech_recognition_service.py` | `SpeechRecognitionService` | Transcribes audio utterances using `faster-whisper` (`tiny.en` int8 CPU) to verify anti-replay challenge phrases. |

---

## 5. API Endpoints & Route Controllers (`api/routes/` and `backend/app/routes/`)

| Route Path | Method | Controller File | Purpose | Key Inputs | Key Output Attributes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `/api/auth/register` | `POST` | `backend/app/routes/auth.py` | Creates a new user account | Username, email, password | User ID, status |
| `/api/auth/login` | `POST` | `backend/app/routes/auth.py` | Authenticates with password | Email/username, password | Bearer JWT access token |
| `/api/auth/enroll-face` | `POST` | `backend/app/routes/auth.py` | Enrolls facial template | User ID, face image file | Template ID, encryption status |
| `/api/v1/face/enroll` | `POST` | `api/routes/face.py` | Enrolls facial biometric template | Base64 image, user ID | AES-GCM ciphertext, status |
| `/api/v1/face/verify` | `POST` | `api/routes/face.py` | Module 1 Face Identity verification | Probe image, enrolled user ID | Match score, verdict, latency |
| `/api/v1/liveness/verify` | `POST` | `api/routes/liveness.py` | Module 2 Face Liveness verification | Frame sequence or single frame | Liveness score, spoof probability, verdict |
| `/api/v1/voice/session/start` | `POST` | `api/routes/voice.py` | Issues challenge phrase for voice auth | Enrolled user ID | Session ID, prompt text, challenge tokens |
| `/api/v1/voice/verify` | `POST` | `api/routes/voice.py` | Module 3 Voice Identity & PAD verification | Audio PCM payload, user ID | Speaker similarity, PAD score, phrase status, verdict |
| `/api/v1/deepfake/analyze-frame` | `POST` | `api/routes/deepfake.py` | Module 4 Single-frame deepfake check | Single facial crop image | Fake probability, verdict, PQC record |
| `/api/v1/deepfake/analyze-stream` | `POST` | `api/routes/deepfake.py` | Module 4 Video stream deepfake check | Video file / frame batch | Temporal fake probability, PQC signature |
| `/api/v1/authenticate/multimodal` | `POST` | `api/routes/multimodal.py` | **Module 5 Unified Multimodal Auth** | Face frame, audio payload, user ID | Final verdict, authorization, risk level, assurance |

---

## 6. Frontend User Interface Architecture (`frontend/src/`)

| File Path | Component Name | Responsibility & Rendered Features |
| :--- | :--- | :--- |
| `frontend/src/App.jsx` | `App` | Master application router, navigation bar with status indicators, responsive layout shell. |
| `frontend/src/context/AuthContext.jsx` | `AuthProvider` | Manages active JWT tokens, session user metadata, login/logout transitions, persistent auth storage. |
| `frontend/src/services/api.js` | Axios Client / API SDK | Unified Axios client with interceptors for bearer authentication tokens and structured error handling. |
| `frontend/src/pages/DashboardPage.jsx` | `DashboardPage` | Executive system status dashboard displaying active modules, biometric status, and quick-action links. |
| `frontend/src/pages/LoginPage.jsx` | `LoginPage` | Standard credentials login interface with routing to biometric step-up authentication. |
| `frontend/src/pages/RegisterPage.jsx` | `RegisterPage` | User registration console initiating subsequent biometric enrollment stages. |
| `frontend/src/pages/FaceEnrollmentPage.jsx` | `FaceEnrollmentPage` | Interactive camera capture guiding the user through high-quality facial enrollment. |
| `frontend/src/pages/VoiceEnrollmentPage.jsx` | `VoiceEnrollmentPage` | Interactive microphone capture recording baseline speaker voice embeddings. |
| `frontend/src/pages/HistoryPage.jsx` | `HistoryPage` | Audit log browser displaying previous verification attempts, risk scores, and PQC signature fingerprints. |
| `frontend/src/pages/SecurityPage.jsx` | `SecurityPage` | Technical overview of post-quantum cryptography parameters, NIST compliance, and encryption metrics. |
| `frontend/src/components/MultimodalAuthConsole.jsx` | `MultimodalAuthConsole` | Comprehensive authentication console synchronizing camera, microphone, real-time diagnostics, and Module 5 decisions. |
| `frontend/src/components/FaceIdentityConsole.jsx` | `FaceIdentityConsole` | Dedicated diagnostic console for Module 1 Face Identity analysis. |
| `frontend/src/components/FaceLivenessConsole.jsx` | `FaceLivenessConsole` | Dedicated diagnostic console for Module 2 Face Liveness passive/temporal analysis. |
| `frontend/src/components/VoiceAuthConsole.jsx` | `VoiceAuthConsole` | Dedicated diagnostic console for Module 3 Voice Identity, PAD, and challenge phrase speech recognition. |
| `frontend/src/components/WebcamCapture.jsx` | `WebcamCapture` | Low-latency WebRTC video stream handler providing frame capture callbacks for inference. |
| `frontend/src/components/MicCapture.jsx` | `MicCapture` | Browser Web Audio API recorder capturing 16 kHz uncompressed PCM audio chunks. |
| `frontend/src/components/RiskGauge.jsx` | `RiskGauge` | Animated radial gauge displaying calculated authentication risk percentage ($0\% - 100\%$). |
| `frontend/src/components/ExplanationCard.jsx` | `ExplanationCard` | Rendered forensic evidence summary breaking down exact metrics from Modules 1–4 and Decision Basis. |
| `frontend/src/components/DeveloperDiagnostics.jsx` | `DeveloperDiagnostics` | Raw JSON telemetry and latency breakdown inspector for system administrators and evaluators. |

---

## 7. Video and Image Preprocessing (`video/`)

| File Path | Key Class / Method | Functionality |
| :--- | :--- | :--- |
| `video/face_detector.py` | `FaceDetector` | Locates faces in raw RGB frames using OpenCV Haar cascades or MTCNN; extracts aligned $160\times160$ and $224\times224$ crops. |
| `video/frame_extractor.py` | `FrameExtractor` | Decodes uploaded MP4, WebM, or AVI video containers at configurable sampling rates (e.g., 5 fps). |
| `video/visualizer.py` | `ForensicVisualizer` | Generates visual diagnostic overlays (bounding boxes, liveness scores, heatmaps) on forensic video evidence. |

---

## 8. Verification & Test Suite (`tests/`)

| Test Suite Path | Target Module / Component | Covered Scenarios |
| :--- | :--- | :--- |
| `tests/voice/test_voice_auth.py` | Module 3: Voice Identity & PAD | 24 passing tests: ECAPA-TDNN embedding shapes (192-D), speaker similarity matching, spoof audio rejection, MFCC extraction, silence handling. |
| `tests/multimodal/test_multimodal_fusion.py` | Module 5: Multimodal Fusion Engine | Fusion decision logic, risk weight calculation, threshold evaluation, incomplete evidence handling. |
| `tests/multimodal/test_multimodal_integration.py` | Module 5: Hard Security Gates | 17 passing tests: Face mismatch gating, spoof rejection gating, deepfake gating, voice mismatch gating, replay gating. |
| `tests/face/test_face_recognition.py` | Module 1: Face Identity | InceptionResnetV1 embedding extraction, $L_2$ normalization, cosine similarity threshold verification ($\tau=0.65$). |
| `tests/face/test_face_liveness.py` | Module 2: Face Liveness Detection | PassiveAntiSpoofNet spatial classification, multi-domain feature extraction (LBP, FFT, Chroma), temporal stability. |
| `tests/deepfake/test_deepqshield.py` | Module 4: DeepQShield Detector | ResNeXt backbone feature extraction, LWE perturbation layer, attention gating, end-to-end classification head. |
| `tests/security/test_pqc.py` | PQC & Biometric Crypto | Kyber-1024 encapsulation/decapsulation, Dilithium-5 sign/verify, AES-256-GCM AEAD template encryption/decryption round-trip. |
