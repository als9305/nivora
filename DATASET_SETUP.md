# Deepfake Dataset Setup Guide (DFDC & DeepGuard Format)

This document explains how to set up the **Deepfake Detection Challenge (DFDC)** dataset or custom deepfake image/video datasets for training and evaluating the **DeepQShield** model.

---

## Expected Folder Structure

```text
data/
├── train/
│   ├── real/
│   │   ├── frame_0001.jpg
│   │   └── ...
│   └── fake/
│       ├── frame_0001.jpg
│       └── ...
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

---

## Step-by-Step Setup

1. **Obtain DFDC Dataset:** Download the official DFDC dataset from Kaggle or official competition sources.
2. **Frame Extraction:** Extract face crops (224×224) using OpenCV or landmark detectors.
3. **Directory Placement:** Place real face crops under `data/train/real/` and manipulated deepfake face crops under `data/train/fake/`.
4. **Validation Split:** Allocate 15-20% of samples into `data/val/real/` and `data/val/fake/`.
5. **Git Notice:** Never commit raw dataset image files to Git repository `.gitignore` rules protect `data/raw/` and `data/processed/`.
