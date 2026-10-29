# DeepGuard Development & Testing Guide

This document describes how to execute local development servers and test suites.

---

## 1. Running Automated Tests

```bash
# Run backend pytest suite
$env:PYTHONPATH="backend;."; python -m pytest backend/tests

# Run DeepQShield API tests
$env:PYTHONPATH="backend;."; python -m pytest tests/api/test_deepfake_api.py

# Run PQC cryptographic tests
python scratch/test_pqc_unit.py

# Run model pipeline unit tests
python scratch/test_models_unit.py
```

---

## 2. Running Local Services

```bash
# Start FastAPI backend server
python -m uvicorn backend.app.main:app --reload --port 8000

# Start React Vite frontend
cd frontend
npm run dev
```
