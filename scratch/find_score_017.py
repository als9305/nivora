import os
import sys
import numpy as np
import torch
import scipy.signal
import huggingface_hub
import pandas as pd

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.voice_auth import voice_auth_engine, AudioPreprocessor, AudioFeatureExtractor

# Let's inspect the weights of net.0 in trained_pad_model_v3.pt
model = voice_auth_engine.anti_spoof_net
w0 = model.net[0].weight.detach().numpy() # [128, 64]
print("Shape of w0:", w0.shape)

# Feature importance: sum of absolute weights connecting each of the 64 features to the 128 hidden neurons
feat_importance = np.sum(np.abs(w0), axis=0) # [64]
print("\nTop 15 most influential feature indices:")
top_indices = np.argsort(feat_importance)[::-1][:15]
feat_names = [
    # 0..12: mfcc_mean (13)
    *[f"mfcc_mean_{i}" for i in range(13)],
    # 13..25: mfcc_std (13)
    *[f"mfcc_std_{i}" for i in range(13)],
    # 26, 27: centroid
    "centroid_mean", "centroid_std",
    # 28, 29: rolloff, flux
    "rolloff_mean", "flux_mean",
    # 30, 31, 32: zcr, f0_mean, f0_std
    "zcr", "f0_mean", "f0_std",
    # 33: hf_ratio
    "hf_ratio",
    # 34..57: log_mel_24_mean (24)
    *[f"log_mel_24_mean_{i}" for i in range(24)],
    # 58: log_mel_24_std
    "log_mel_24_std",
    # 59..63: padding
    *[f"pad_{i}" for i in range(5)]
]

for idx in top_indices:
    name = feat_names[idx] if idx < len(feat_names) else f"feat_{idx}"
    print(f"  Index {idx:2d} ({name:20s}): importance = {feat_importance[idx]:.4f}")

# Now, let's load ASVspoof 2017 authentic vs spoof and print average feature values:
p17 = huggingface_hub.hf_hub_download(
    repo_id="DynamicSuperb/SpoofDetection_ASVspoof2017",
    filename="data/test-00000-of-00001-e7ed034c56ededdd.parquet",
    repo_type="dataset"
)
df17 = pd.read_parquet(p17)

bona_feats = []
for _, row in df17[df17['label'] == 'authentic'].head(20).iterrows():
    sig, sr, _ = AudioPreprocessor.decode_wav_bytes(row['audio']['bytes'])
    bona_feats.append(AudioFeatureExtractor.extract_features(sig, sr)['feature_vector'])

spoof_feats = []
for _, row in df17[df17['label'] == 'spoofed'].head(20).iterrows():
    sig, sr, _ = AudioPreprocessor.decode_wav_bytes(row['audio']['bytes'])
    spoof_feats.append(AudioFeatureExtractor.extract_features(sig, sr)['feature_vector'])

bona_arr = np.array(bona_feats)
spoof_arr = np.array(spoof_feats)

print("\n--- Key feature differences between BONAFIDE and SPOOF (ASVspoof 2017) ---")
for idx in top_indices[:10]:
    name = feat_names[idx] if idx < len(feat_names) else f"feat_{idx}"
    b_mean = np.mean(bona_arr[:, idx])
    s_mean = np.mean(spoof_arr[:, idx])
    print(f"  Index {idx:2d} ({name:20s}): BONAFIDE mean={b_mean:10.4f} | SPOOF mean={s_mean:10.4f}")
