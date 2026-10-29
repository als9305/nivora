import os
import sys
root_dir = os.path.abspath(os.path.dirname(__file__) + "/..")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
import glob
import torch
import torch.nn.functional as F
import numpy as np

from models.voice_auth import voice_auth_engine, AudioPreprocessor, AudioFeatureExtractor

print(f"=== CHECKPOINT SHA256: {voice_auth_engine.anti_spoof_net.checkpoint_sha256} ===")
print(f"=== TAU_LIVENESS: {voice_auth_engine.tau_liveness}, TAU_SPEAKER: {voice_auth_engine.tau_speaker} ===")

print("\n--- REAL SPEECH SAMPLES ---")
for p in sorted(glob.glob("scratch/real_speech/*.wav")):
    with open(p, "rb") as f:
        data = f.read()
    sig, sr, audit = AudioPreprocessor.decode_wav_bytes(data)
    feats = AudioFeatureExtractor.extract_features(sig, sr)
    fvec = feats["feature_vector"]
    ac = feats["acoustic_metrics"]
    
    with torch.no_grad():
        t = torch.from_numpy(fvec).float().unsqueeze(0)
        logits = voice_auth_engine.anti_spoof_net(t)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
    score, spoof_p, att = voice_auth_engine.evaluate_anti_spoof(fvec, ac)
    res = voice_auth_engine.process_voice_pipeline(data)
    print(f"{os.path.basename(p):12s} | NN_live: {probs[0]:.4f} NN_spoof: {probs[1]:.4f} | eval_score: {score:.4f} spoof: {spoof_p:.4f} att: {att} | pipeline_score: {res['liveness_score']} verd: {res['liveness_verdict']} | hf_ratio: {ac.get('high_frequency_ratio', 0.0):.4f} zcr: {ac.get('zero_crossing_rate', 0.0):.4f} f0_std: {ac.get('f0_std_hz', 0.0):.4f}")

print("\n--- FAKE/SPOOF SAMPLES ---")
for p in sorted(glob.glob("datasets/voice_fake/*.wav")):
    with open(p, "rb") as f:
        data = f.read()
    sig, sr, audit = AudioPreprocessor.decode_wav_bytes(data)
    feats = AudioFeatureExtractor.extract_features(sig, sr)
    fvec = feats["feature_vector"]
    ac = feats["acoustic_metrics"]
    
    with torch.no_grad():
        t = torch.from_numpy(fvec).float().unsqueeze(0)
        logits = voice_auth_engine.anti_spoof_net(t)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
    score, spoof_p, att = voice_auth_engine.evaluate_anti_spoof(fvec, ac)
    res = voice_auth_engine.process_voice_pipeline(data)
    print(f"{os.path.basename(p):22s} | NN_live: {probs[0]:.4f} NN_spoof: {probs[1]:.4f} | eval_score: {score:.4f} spoof: {spoof_p:.4f} att: {att} | pipeline_score: {res['liveness_score']} verd: {res['liveness_verdict']} | hf_ratio: {ac.get('high_frequency_ratio', 0.0):.4f} zcr: {ac.get('zero_crossing_rate', 0.0):.4f} f0_std: {ac.get('f0_std_hz', 0.0):.4f}")
