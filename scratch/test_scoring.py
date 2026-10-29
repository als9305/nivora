import os
import sys
import numpy as np
import scipy.signal
import torch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.voice_auth import voice_auth_engine, AudioPreprocessor, AudioFeatureExtractor
from scripts.benchmark_voice_auth import generate_benchmark_speech_sample

def test_scoring_trace():
    s1 = generate_benchmark_speech_sample(1.2, base_f0=120.0, attack_type="GENUINE_LIVE")
    s2 = generate_benchmark_speech_sample(1.2, base_f0=240.0, attack_type="GENUINE_LIVE")

    sig1, _, _ = AudioPreprocessor.decode_wav_bytes(s1)
    sig2, _, _ = AudioPreprocessor.decode_wav_bytes(s2)

    feats1 = AudioFeatureExtractor.extract_features(sig1)
    feats2 = AudioFeatureExtractor.extract_features(sig2)

    lm1 = feats1["log_mel_80"]
    lm2 = feats2["log_mel_80"]

    # Without CMVN
    emb1_raw = voice_auth_engine.extract_embedding(lm1)
    emb2_raw = voice_auth_engine.extract_embedding(lm2)
    cos_raw = voice_auth_engine.compute_similarity(emb1_raw, emb2_raw)

    # With CMVN
    lm1_cmvn = lm1 - np.mean(lm1, axis=1, keepdims=True)
    lm2_cmvn = lm2 - np.mean(lm2, axis=1, keepdims=True)
    emb1_cmvn = voice_auth_engine.extract_embedding(lm1_cmvn)
    emb2_cmvn = voice_auth_engine.extract_embedding(lm2_cmvn)
    cos_cmvn = voice_auth_engine.compute_similarity(emb1_cmvn, emb2_cmvn)

    print(f"RAW Cosine Similarity (120Hz vs 240Hz):  {cos_raw:.6f}")
    print(f"CMVN Cosine Similarity (120Hz vs 240Hz): {cos_cmvn:.6f}")
    print("\nSample embedding 1 (first 10 elements):", emb1_cmvn[:10])
    print("Sample embedding 2 (first 10 elements):", emb2_cmvn[:10])

if __name__ == "__main__":
    test_scoring_trace()
