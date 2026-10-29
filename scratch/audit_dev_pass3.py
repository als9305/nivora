import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import huggingface_hub

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.voice_auth import AudioPreprocessor, AudioFeatureExtractor, PassiveAudioAntiSpoofNet, voice_auth_engine

print("=" * 70)
print("MODULE 4 — STEP 1 & 2 & 3: DEV DATASET & SCORE DISTRIBUTION AUDIT")
print("=" * 70)

# 1. Fetch DEV samples
cat_samples = {
    'BONAFIDE': [],
    'REPLAY_ATTACK': [],
    'SYNTHETIC_TTS': [],
    'VOICE_CLONE': []
}

try:
    p17 = huggingface_hub.hf_hub_download(
        repo_id="DynamicSuperb/SpoofDetection_ASVspoof2017",
        filename="data/test-00000-of-00001-e7ed034c56ededdd.parquet",
        repo_type="dataset"
    )
    df17 = pd.read_parquet(p17)
    for _, row in df17.iterrows():
        b = row['audio']['bytes']
        if row['label'] == 'authentic':
            cat_samples['BONAFIDE'].append(b)
        elif row['label'] == 'spoofed':
            cat_samples['REPLAY_ATTACK'].append(b)
except Exception as e:
    print(f"Error loading ASVspoof 2017: {e}")

try:
    p15 = huggingface_hub.hf_hub_download(
        repo_id="DynamicSuperb/SpoofDetection_ASVspoof2015",
        filename="data/test-00000-of-00001-4a252b1e92dc86af.parquet",
        repo_type="dataset"
    )
    df15 = pd.read_parquet(p15)
    for i, row in df15.iterrows():
        b = row['audio']['bytes']
        if row['label'] == 'authentic':
            cat_samples['BONAFIDE'].append(b)
        elif row['label'] == 'spoofed':
            if i % 2 == 0:
                cat_samples['SYNTHETIC_TTS'].append(b)
            else:
                cat_samples['VOICE_CLONE'].append(b)
except Exception as e:
    print(f"Error loading ASVspoof 2015: {e}")

# 2. Extract DEV features
dev_bonafide_scores = []
dev_spoof_scores = []

for cat, samples in cat_samples.items():
    n_dev = int(len(samples) * 0.4)
    dev_subset = samples[:n_dev]

    print(f"Processing DEV '{cat}': {len(dev_subset)} samples...")
    for s in dev_subset:
        sig, sr, audit = AudioPreprocessor.decode_wav_bytes(s)
        if sig is not None and len(sig) > 1000:
            res = voice_auth_engine.process_voice_pipeline(s)
            score = res['liveness_score']
            if cat == 'BONAFIDE':
                dev_bonafide_scores.append(score)
            else:
                dev_spoof_scores.append(score)

dev_b = np.array(dev_bonafide_scores)
dev_s = np.array(dev_spoof_scores)

print("\n" + "-" * 50)
print("DEV SPLIT SCORE STATISTICAL AUDIT (Liveness Score = p_live):")
print("-" * 50)
print(f"BONAFIDE (n={len(dev_b)}):")
print(f"  min:    {np.min(dev_b):.4f}")
print(f"  max:    {np.max(dev_b):.4f}")
print(f"  mean:   {np.mean(dev_b):.4f}")
print(f"  median: {np.median(dev_b):.4f}")
print(f"  std:    {np.std(dev_b):.4f}")

print(f"\nSPOOF (n={len(dev_s)}):")
print(f"  min:    {np.min(dev_s):.4f}")
print(f"  max:    {np.max(dev_s):.4f}")
print(f"  mean:   {np.mean(dev_s):.4f}")
print(f"  median: {np.median(dev_s):.4f}")
print(f"  std:    {np.std(dev_s):.4f}")

# Check overlap
overlap_min = max(np.min(dev_b), np.min(dev_s))
overlap_max = min(np.max(dev_b), np.max(dev_s))
print(f"\nScore Range Overlap: [{overlap_min:.4f}, {overlap_max:.4f}]")
