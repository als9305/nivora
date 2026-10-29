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
print("MODULE 4 — PASS 4 STEP 1 & 2 & 3: DATA PIPELINE & MODEL AUDIT")
print("=" * 70)

# Fetch samples
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

dev_data = {}
test_data = {}

for cat, samples in cat_samples.items():
    n_dev = int(len(samples) * 0.4)
    dev_data[cat] = samples[:n_dev]
    test_data[cat] = samples[n_dev:]

print("\n--- SAMPLE COUNTS BY SPLIT & ATTACK CATEGORY ---")
print("DEV SPLIT (40%):")
for cat, s in dev_data.items():
    print(f"  {cat}: {len(s)} samples")

print("HELD-OUT TEST SPLIT (60%):")
for cat, s in test_data.items():
    print(f"  {cat}: {len(s)} samples")

# Evaluate score distributions for DEV and TEST separately
def calc_stats(scores_list):
    arr = np.array(scores_list)
    return {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr))
    }

print("\n--- DEV SCORE STATISTICAL BREAKDOWN ---")
for cat, samples in dev_data.items():
    scores = []
    for s in samples:
        sig, sr, audit = AudioPreprocessor.decode_wav_bytes(s)
        if sig is not None and len(sig) > 1000:
            res = voice_auth_engine.process_voice_pipeline(s)
            scores.append(res['liveness_score'])
    st = calc_stats(scores)
    print(f"{cat}: mean={st['mean']:.4f}, std={st['std']:.4f}, min={st['min']:.4f}, max={st['max']:.4f}, median={st['median']:.4f}")

print("\n--- TEST SCORE STATISTICAL BREAKDOWN ---")
for cat, samples in test_data.items():
    scores = []
    for s in samples:
        sig, sr, audit = AudioPreprocessor.decode_wav_bytes(s)
        if sig is not None and len(sig) > 1000:
            res = voice_auth_engine.process_voice_pipeline(s)
            scores.append(res['liveness_score'])
    st = calc_stats(scores)
    print(f"{cat}: mean={st['mean']:.4f}, std={st['std']:.4f}, min={st['min']:.4f}, max={st['max']:.4f}, median={st['median']:.4f}")
