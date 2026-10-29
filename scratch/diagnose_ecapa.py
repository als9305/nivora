import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.signal
import io
import wave
from huggingface_hub import hf_hub_download

# Add root dir to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.voice_auth import voice_auth_engine, AudioPreprocessor, AudioFeatureExtractor
from scripts.benchmark_voice_auth import generate_benchmark_speech_sample

def run_diagnostic():
    print("=" * 65)
    print("DIAGNOSTIC PASS 2 — SCIENTIFIC TRACE OF ECAPA-TDNN PIPELINE")
    print("=" * 65)

    # 1. Inspect Pretrained Model Weights State
    print("\n1. MODEL PRETRAINED WEIGHTS CHECK:")
    print(f"   Model Name: {voice_auth_engine.embedding_net.model_info['model_name']}")
    print(f"   Pretrained Weights Loaded: {voice_auth_engine.embedding_net.is_pretrained}")
    print(f"   Weight Source: {voice_auth_engine.embedding_net.model_info['weight_source']}")
    print(f"   Embedding Dim: {voice_auth_engine.embedding_net.embedding_dim}")

    # 2. Test Audio Samples
    spk1_s1 = generate_benchmark_speech_sample(1.2, base_f0=120.0, f0_std=14.0, attack_type="GENUINE_LIVE")
    spk1_s2 = generate_synthetic_wav(1.2, base_f0=121.5, f0_std=13.5, attack_type="GENUINE_LIVE")
    spk2_s1 = generate_benchmark_speech_sample(1.2, base_f0=240.0, f0_std=20.0, attack_type="GENUINE_LIVE")

    sig1_1, sr1, _ = AudioPreprocessor.decode_wav_bytes(spk1_s1)
    sig1_2, sr2, _ = AudioPreprocessor.decode_wav_bytes(spk1_s2)
    sig2_1, sr3, _ = AudioPreprocessor.decode_wav_bytes(spk2_s1)

    feats1_1 = AudioFeatureExtractor.extract_features(sig1_1, sr1)
    feats1_2 = AudioFeatureExtractor.extract_features(sig1_2, sr2)
    feats2_1 = AudioFeatureExtractor.extract_features(sig2_1, sr3)

    log_mel1_1 = feats1_1["log_mel_80"]
    log_mel1_2 = feats1_2["log_mel_80"]
    log_mel2_1 = feats2_1["log_mel_80"]

    print("\n2. UNNORMALIZED LOG-MEL EMBEDDINGS (Current Implementation):")
    emb1_1_raw = voice_auth_engine.extract_embedding(log_mel1_1)
    emb1_2_raw = voice_auth_engine.extract_embedding(log_mel1_2)
    emb2_1_raw = voice_auth_engine.extract_embedding(log_mel2_1)

    sim_genuine_raw = voice_auth_engine.compute_similarity(emb1_1_raw, emb1_2_raw)
    sim_impostor_raw = voice_auth_engine.compute_similarity(emb1_1_raw, emb2_1_raw)

    print(f"   Genuine Pair Cosine Similarity (Raw):   {sim_genuine_raw:.6f}")
    print(f"   Impostor Pair Cosine Similarity (Raw):  {sim_impostor_raw:.6f}")
    print(f"   Score Gap (Genuine - Impostor):         {sim_genuine_raw - sim_impostor_raw:.6f}")

    # 3. Test CMVN Normalized Log-Mel (SpeechBrain Standard)
    print("\n3. CMVN NORMALIZED LOG-MEL EMBEDDINGS (SpeechBrain Standard):")
    log_mel1_1_cmvn = log_mel1_1 - np.mean(log_mel1_1, axis=1, keepdims=True)
    log_mel1_2_cmvn = log_mel1_2 - np.mean(log_mel1_2, axis=1, keepdims=True)
    log_mel2_1_cmvn = log_mel2_1 - np.mean(log_mel2_1, axis=1, keepdims=True)

    emb1_1_cmvn = voice_auth_engine.extract_embedding(log_mel1_1_cmvn)
    emb1_2_cmvn = voice_auth_engine.extract_embedding(log_mel1_2_cmvn)
    emb2_1_cmvn = voice_auth_engine.extract_embedding(log_mel2_1_cmvn)

    sim_genuine_cmvn = voice_auth_engine.compute_similarity(emb1_1_cmvn, emb1_2_cmvn)
    sim_impostor_cmvn = voice_auth_engine.compute_similarity(emb1_1_cmvn, emb2_1_cmvn)

    print(f"   Genuine Pair Cosine Similarity (CMVN):  {sim_genuine_cmvn:.6f}")
    print(f"   Impostor Pair Cosine Similarity (CMVN): {sim_impostor_cmvn:.6f}")
    print(f"   Score Gap (Genuine - Impostor):         {sim_genuine_cmvn - sim_impostor_cmvn:.6f}")

def generate_synthetic_wav(duration_sec=1.2, sample_rate=16000, base_f0=180.0, f0_std=15.0, attack_type="GENUINE_LIVE"):
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    f0_contour = base_f0 + f0_std * np.sin(2 * np.pi * 3.0 * t)
    phase = 2 * np.pi * np.cumsum(f0_contour) / sample_rate
    signal = 0.5 * np.sin(phase) + 0.25 * np.sin(2 * phase)
    pcm_data = (np.clip(signal, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data.tobytes())
    return buf.getvalue()

if __name__ == "__main__":
    run_diagnostic()
