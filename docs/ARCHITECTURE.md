# DeepGuard System Architecture

```text
User / Web Browser
   |
   | (WebRTC Video & Audio Streams)
   v
React Frontend (Vite + Tailwind CSS + Recharts)
   |
   | (REST APIs over HTTPS/JSON)
   v
FastAPI Backend (Uvicorn / Python 3.13)
   |
   +----> Face Verification Engine (OpenCV HOG / Spatial Embeddings)
   |
   +----> Active Liveness Challenge Engine (MediaPipe / Landmark displacement)
   |
   +----> Deepfake Detector (Spatial-Temporal Frequency & Edge Artifact Adapter)
   |
   +----> Voice Verification & Anti-Spoofing (MFCCs & Pitch Monotonicity Analysis)
   |
   +----> Device & Session Risk Engine (User-Agent, Rate-limiting, Fingerprints)
   |
   v
Deterministic Risk Engine (Weighted Scoring 0 - 100)
   |
   +----> LOW (0 - 35)    -> ALLOW
   |
   +----> MEDIUM (36 - 70) -> STEP-UP (OTP)
   |
   +----> HIGH (71 - 100) -> BLOCK
   |
   v
Explainable Security Agent (Evidence Report & LLM Threat Analysis)
   |
   v
Security Incident Dashboard & Audit History (MongoDB / In-Memory Store)
```

## Security Core Principle
Authentication decisions are **never** made using face recognition alone. A multi-modal composite risk score is evaluated across independent threat vectors to guarantee resilience against deepfakes and presentation replay attacks.
