import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import huggingface_hub
import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.voice_auth import voice_auth_engine, AudioPreprocessor, AudioFeatureExtractor

p17 = huggingface_hub.hf_hub_download(
    repo_id="DynamicSuperb/SpoofDetection_ASVspoof2017",
    filename="data/test-00000-of-00001-e7ed034c56ededdd.parquet",
    repo_type="dataset"
)
df = pd.read_parquet(p17)

auth_bytes = df[df['label'] == 'authentic'].iloc[0]['audio']['bytes']
spoof_bytes = df[df['label'] == 'spoofed'].iloc[0]['audio']['bytes']

sig_a, sr_a, _ = AudioPreprocessor.decode_wav_bytes(auth_bytes)
feats_a = AudioFeatureExtractor.extract_features(sig_a, sr_a)

sig_s, sr_s, _ = AudioPreprocessor.decode_wav_bytes(spoof_bytes)
feats_s = AudioFeatureExtractor.extract_features(sig_s, sr_s)

with torch.no_grad():
    vec_a = torch.from_numpy(feats_a['feature_vector']).float().unsqueeze(0)
    logits_a = voice_auth_engine.anti_spoof_net(vec_a)
    probs_a = F.softmax(logits_a, dim=1).squeeze(0).cpu().numpy()

    vec_s = torch.from_numpy(feats_s['feature_vector']).float().unsqueeze(0)
    logits_s = voice_auth_engine.anti_spoof_net(vec_s)
    probs_s = F.softmax(logits_s, dim=1).squeeze(0).cpu().numpy()

print(f"AUTHENTIC Logits: {logits_a.numpy()}, Probs: index 0={probs_a[0]:.4f}, index 1={probs_a[1]:.4f}")
print(f"SPOOFED Logits:   {logits_s.numpy()}, Probs: index 0={probs_s[0]:.4f}, index 1={probs_s[1]:.4f}")
