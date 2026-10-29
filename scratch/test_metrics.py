import os
import sys
sys.path.insert(0, ".")
from models.voice_auth import AudioPreprocessor, AudioFeatureExtractor, voice_auth_engine

for i in range(1, 9):
    path = f"scratch/real_speech/real_{i}.wav"
    with open(path, "rb") as f:
        data = f.read()
    sig, sr, audit = AudioPreprocessor.decode_wav_bytes(data)
    feats = AudioFeatureExtractor.extract_features(sig, sr)
    m = feats["acoustic_metrics"]
    res = voice_auth_engine.process_voice_pipeline(data)
    p = res["liveness_score"]
    print(f"real_{i}.wav (p={p:.4f}): centroid={m['spectral_centroid_hz']} rolloff={m['spectral_rolloff_hz']} zcr={m['zero_crossing_rate']} f0_m={m['f0_mean_hz']} f0_s={m['f0_std_hz']} hf_ratio={m['high_frequency_ratio']}")
