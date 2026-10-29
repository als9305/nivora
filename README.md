# DeepGuard — Deepfake-Resilient Adaptive Multi-Modal Authentication Framework

**DeepGuard** is a working cybersecurity prototype demonstrating that traditional single-factor face recognition can be easily bypassed by AI deepfakes, photo replays, and synthetic voice clones. DeepGuard proves that authentication resilience is dramatically enhanced when multiple independent biometric, spatial-temporal, acoustic, and device signals are combined into a deterministic risk engine.

---

## Key Features

1. **Multi-Modal Biometric Fusion**:
   - **Face Identity Verification**: Pretrained embedding similarity.
   - **Active Liveness Detection**: Challenge-response verification (BLINK, TURN_LEFT, TURN_RIGHT, SMILE, NOD) preventing static photo replay attacks.
   - **Deepfake / Synthetic Media Inspection**: Multi-frame spatial-temporal frequency analysis (FFT artifacts, color gamut boundary inconsistencies, and residual variance).
   - **Voice Verification & Anti-Spoofing**: Audio MFCC feature matching, speech detection, and pitch-monotone synthesis detection.
   - **Device & Session Risk Engine**: Detection of headless browser automation, rapid brute force, and untrusted device fingerprints.
2. **Deterministic Risk Engine**:
   - Configurable weighted score (0–100 scale):
     - **LOW (0 – 35)**: `ALLOW` (Access Granted)
     - **MEDIUM (36 – 70)**: `STEP-UP` (OTP Verification Required)
     - **HIGH (71 – 100)**: `BLOCK` (Access Blocked)
3. **Explainable Security Decision Agent**:
   - Human-readable threat reports, evidence breakdown, and AI analyst recommendations.
4. **Hackathon Security Attack Simulator**:
   - Live scenario testing for `GENUINE_USER`, `PHOTO_ATTACK`, `REPLAY_ATTACK`, `DEEPFAKE_ATTACK`, `VOICE_SPOOF`, and `BORDERLINE_CASE`.
5. **Zero-Config Database Flexibility**:
   - Standard MongoDB support + automatic fallback to an in-memory repository if MongoDB is disconnected.

---

## Technology Stack

- **Frontend**: React, Vite, JavaScript, Tailwind CSS, Recharts, Axios, React Router, WebRTC/MediaDevices API.
- **Backend**: Python 3.13, FastAPI, Uvicorn, Pydantic v2, PyJWT, Passlib/Bcrypt.
- **Computer Vision & ML**: OpenCV, NumPy, Pillow, SciPy.
- **Database**: MongoDB (Motor / PyMongo) with In-Memory fallback.

---

## Project Structure

```text
deepguard/
│
├── frontend/             # React + Vite + Tailwind CSS Application
│   ├── src/
│   │   ├── components/   # WebcamCapture, MicCapture, RiskGauge, ExplanationCard, Navbar
│   │   ├── pages/        # RegisterPage, LoginPage, DashboardPage, SimulationPage, HistoryPage, IncidentDetailPage
│   │   ├── services/     # api.js REST API Client
│   │   ├── hooks/        # useWebcam.js, useAudioRecorder.js
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── backend/              # FastAPI Python Backend
│   ├── app/
│   │   ├── main.py       # Server entrypoint & lifecycle
│   │   ├── config.py     # Risk weights & model thresholds
│   │   ├── routes/       # auth.py, verification.py, dashboard.py, health.py
│   │   ├── services/     # face_service, liveness_service, deepfake_service, voice_service, risk_engine
│   │   ├── ml/           # face_model, deepfake_model, voice_model, model_loader
│   │   ├── database/     # mongodb.py, repositories.py
│   │   ├── schemas/      # Pydantic data schemas
│   │   └── utils/        # security, files, logging
│   ├── tests/            # Pytest suite
│   ├── requirements.txt
│   └── .env.example
│
├── models/               # Model weights documentation
├── datasets/             # Sample datasets documentation
├── docs/                 # ARCHITECTURE.md, API.md, SECURITY.md, DEMO.md
├── docker-compose.yml
└── README.md
```

---

## Quick Start (Local Setup)

### 1. Run FastAPI Backend

```bash
cd backend
python -m pip install -r requirements.txt
set PYTHONPATH=.
python -m uvicorn app.main:app --reload --port 8000
```

Backend API will start at `http://localhost:8000` (Health Check: `http://localhost:8000/health`).

### 2. Run React Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend application will open at `http://localhost:5173`.

---

## Running Automated Tests

```bash
cd backend
set PYTHONPATH=.
python -m pytest tests/
```

---

## Demo Scenarios

### Scenario 1 — Genuine Login
- **Inputs**: Real face, completed liveness challenge, real video frame, clean audio phrase.
- **Expected**: `Face: PASS`, `Liveness: PASS`, `Deepfake: LOW`, `Voice: PASS`. Risk = `LOW (0-35)`. Decision = `ALLOW`.

### Scenario 2 — Photo Attack
- **Inputs**: Printed photo held in front of webcam.
- **Expected**: `Face: MATCH`, `Liveness: FAIL (0% motion)`. Risk = `HIGH (>= 71)`. Decision = `BLOCK`.

### Scenario 3 — Deepfake Face Swap
- **Inputs**: AI synthetic face swap video feed.
- **Expected**: `Face: MATCH`, `Deepfake Probability: 93%`. Risk = `HIGH (>= 71)`. Decision = `BLOCK`.

---

## License

MIT License — Hackathon Cybersecurity Prototype.
