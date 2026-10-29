import os
import sys
import huggingface_hub
import pandas as pd
import numpy as np
import torch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.voice_auth import voice_auth_engine, AudioPreprocessor, AudioFeatureExtractor

print("=== TESTING REAL ASVSPOOF AUTHENTIC SPEECH SAMPLES ===")
p17 = huggingface_hub.hf_hub_download(
    repo_id="DynamicSuperb/SpoofDetection_ASVspoof2017",
    filename="data/test-00000-of-00001-e7ed034c56ededdd.parquet",
    repo_type="dataset"
)
df17 = pd.read_parquet(p17)
auth_df = df17[df17['label'] == 'authentic'].head(10)

for idx, row in auth_df.iterrows():
    audio_b = row['audio']['bytes']
    res = voice_auth_engine.process_voice_pipeline(audio_b)
    
    # Also inspect feature extraction directly
    sig, sr, audit = AudioPreprocessor.decode_wav_bytes(audio_b)
    feats = AudioFeatureExtractor.extract_features(sig, sr)
    fvec = feats['feature_vector']
    with torch.no_grad():
        t = torch.from_numpy(fvec).float().unsqueeze(0)
        logits = voice_auth_engine.anti_spoof_net(t)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
    
    print(f"Sample {idx}:")
    print(f"  Neural Net Raw Logits: {logits.numpy().tolist()}")
    print(f"  Neural Net Softmax: p_live={probs[0]:.4f}, p_spoof={probs[1]:.4f}")
    print(f"  Acoustic metrics: {feats['acoustic_metrics']}")
    print(f"  Pipeline Result: score={res['liveness_score']}, verdict={res['liveness_verdict']}, attack={res.get('attack_type')}")
