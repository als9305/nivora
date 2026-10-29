# DeepGuard Installation & Setup Guide

This guide describes how to install and configure the **DeepQShield** deepfake detection subsystem.

---

## Prerequisites

- Python 3.10+ (Python 3.13 supported)
- PyTorch & Torchvision
- Node.js 18+ (for frontend dashboard)

---

## 1. Backend Installation

```bash
# Install required Python packages
python -m pip install -r backend/requirements.txt
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install pycryptodome pyyaml scikit-learn matplotlib pandas timm albumentations
```

---

## 2. Frontend Installation

```bash
cd frontend
npm install
```

---

## 3. Environment Configuration

Copy `.env.example` to `.env`:

```bash
cp backend/.env.example backend/.env
```
