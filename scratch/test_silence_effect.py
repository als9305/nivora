import os
import sys
import numpy as np
import torch
import huggingface_hub
import pandas as pd

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.voice_auth import voice_auth_engine, AudioPreprocessor, AudioFeatureExtractor

p17 = huggingface_hub.hf_hub_download(
    repo_id="DynamicSuperb/SpoofDetection_ASVspoof2017",
    filename="data/test-00000-of-00001-e7ed034c56ededdd.parquet",
    repo_type="dataset"
)
df17 = pd.read_parquet(p17)

for sample_idx in range(5):
    auth_row = df17[df17['label'] == 'authentic'].iloc[sample_idx]
    audio_b = auth_row['audio']['bytes']
    sig, sr, _ = AudioPreprocessor.decode_wav_bytes(audio_b)
    
    # 1. Clean original
    res_clean = voice_auth_engine.process_voice_pipeline(audio_b)
    
    # 2. Add 1.0s silence (zero padding) before and after
    silence = np.zeros(sr, dtype=np.float32)
    sig_padded = np.concatenate([silence, sig, silence])
    # Encode as WAV bytes
    import io, wave
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((sig_padded * 32767).astype(np.int16).tobytes())
    res_padded = voice_auth_engine.process_voice_pipeline(buf.getvalue())
    
    # 3. Add 1.0s room noise / low hiss (ambient mic noise RMS = 0.005) before and after
    noise = np.random.normal(0, 0.005, sr).astype(np.float32)
    sig_noisy = np.concatenate([noise, sig, noise])
    buf_noise = io.BytesIO()
    with wave.open(buf_noise, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((sig_noisy * 32767).astype(np.int16).tobytes())
    res_noisy = voice_auth_engine.process_voice_pipeline(buf_noise.getvalue())
    
    print(f"Sample {sample_idx}:")
    print(f"  Clean:  score={res_clean['liveness_score']:.4f}, verdict={res_clean['liveness_verdict']}, attack={res_clean.get('attack_type')}")
    print(f"  Padded: score={res_padded['liveness_score']:.4f}, verdict={res_padded['liveness_verdict']}, attack={res_padded.get('attack_type')}")
    print(f"  Noisy:  score={res_noisy['liveness_score']:.4f}, verdict={res_noisy['liveness_verdict']}, attack={res_noisy.get('attack_type')}")
