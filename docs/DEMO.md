# DeepGuard Hackathon Demo Script & Judge Guide

This guide details the step-by-step presentation script for demonstrating DeepGuard to hackathon judges.

---

## 1. Executive Summary (30 Seconds)
"Single-factor face authentication is dead. Today, free consumer AI apps can generate real-time deepfakes that fool traditional face recognition. DeepGuard solves this by combining face matching, active liveness challenges, spatial-temporal deepfake detection, voice verification, and device risk into a unified adaptive risk engine."

---

## 2. Live Demo Sequence

### Demo 1: Registration
1. Navigate to `/register`.
2. Enter Name, Email, Password.
3. Allow camera access -> Capture Face Baseline.
4. Record challenge phrase -> Complete Registration.

### Demo 2: Attack Simulation Studio (Crucial for Judges!)
1. Navigate to `/simulation`.
2. Click **"Photo Replay Attack"** -> Observe system output: `Face Match: 92%`, `Liveness: 15% (FAIL)`, `Decision: BLOCK`. Explain to judges why traditional face recognition alone fails.
3. Click **"Deepfake AI Face-Swap"** -> Observe system output: `Face Match: 95%`, `Deepfake Probability: 93%`, `Decision: BLOCK`.
4. Click **"Borderline Identity Anomaly"** -> Observe system output: `Risk Score: 48/100 (MEDIUM)`, `Decision: STEP-UP OTP`.

### Demo 3: Security Dashboard Inspection
1. Navigate to `/dashboard`.
2. Highlight the **Risk Over Time AreaChart**, **Decisions PieChart**, and **Incident Audit Trail**.
3. Click **"Inspect"** on an incident to present the **Explainable AI Decision Report**.
