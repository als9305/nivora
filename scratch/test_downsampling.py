import os
import sys
import numpy as np
import scipy.signal
import torch
import huggingface_hub
import pandas as pd

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.voice_auth import voice_auth_engine, AudioPreprocessor, AudioFeatureExtractor

# Fetch a 48kHz audio or resample a clean audio to 48kHz, then downsample using:
# Method A: Naive decimation (every 3rd sample, as done in encodeWavPcm16)
# Method B: Scipy proper anti-aliased resample (scipy.signal.resample_poly)
p17 = huggingface_hub.hf_hub_download(
    repo_id="DynamicSuperb/SpoofDetection_ASVspoof2017",
    filename="data/test-00000-of-00001-e7ed034c56ededdd.parquet",
    repo_type="dataset"
)
df17 = pd.read_parquet(p17)
auth_row = df17[df17['label'] == 'authentic'].iloc[1]
audio_b = auth_row['audio']['bytes']
sig16, sr, _ = AudioPreprocessor.decode_wav_bytes(audio_b)

# Simulate 48kHz browser recording from this 16kHz speech:
sig48 = scipy.signal.resample_poly(sig16, 3, 1)

# Method A: Naive decimation (like frontend encodeWavPcm16)
sig_naive = sig48[::3]

# Method B: Proper anti-aliased resampling (scipy resample_poly)
sig_poly = scipy.signal.resample_poly(sig48, 1, 3)

# Method C: Original 16kHz
for label, s in [
    ("Original 16kHz", sig16),
    ("Properly Resampled 16kHz", sig_poly),
    ("Naively Decimated 16kHz (frontend)", sig_naive),
]:
    feats = AudioFeatureExtractor.extract_features(s, 16000)
    fvec = feats['feature_vector']
    with torch.no_grad():
        t = torch.from_numpy(fvec).float().unsqueeze(0)
        logits = voice_auth_engine.anti_spoof_net(t)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
    score, spoof, att = voice_auth_engine.evaluate_anti_spoof(fvec, feats['acoustic_metrics'])
    print(f"\n--- {label} ---")
    print(f"  Logits: {logits.numpy().tolist()}")
    print(f"  Softmax: p_live={probs[0]:.4f}, p_spoof={probs[1]:.4f}")
    print(f"  Metrics: {feats['acoustic_metrics']}")
    print(f"  Pipeline score: {score}, verdict: {score >= 0.70}")
