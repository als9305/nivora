import sys, os
sys.path.insert(0, os.path.abspath("."))
import torch
import numpy as np
import io
import wave
from models.voice_auth import PassiveAudioAntiSpoofNet, AudioFeatureExtractor, AudioPreprocessor, voice_auth_engine

pad = voice_auth_engine.anti_spoof_net
print("PAD model info:", pad.model_info)
print("Pretrained:", pad.is_pretrained)
print("Weights path:", pad.model_info["weight_source"])

# Check determinism: 5 consecutive runs on same random tensor
t_det = torch.randn(1, 64)
outputs = []
for i in range(5):
    with torch.inference_mode():
        logits = pad(t_det)
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
        outputs.append(probs)
print("Determinism test (5 runs on same tensor):")
for i, p in enumerate(outputs):
    print(f"Run {i+1}: p_live={p[0]:.8f}, p_spoof={p[1]:.8f}")
diff = max(abs(outputs[0][0] - p[0]) for p in outputs)
print("Max absolute difference across runs:", diff)
assert diff == 0.0, "PAD model is NOT deterministic!"

# Find .wav files
wav_files = []
for root, dirs, files in os.walk('.'):
    if '.venv' in root or 'node_modules' in root:
        continue
    for f in files:
        if f.endswith('.wav'):
            wav_files.append(os.path.join(root, f))

print(f"\nFound {len(wav_files)} WAV files:")
for path in wav_files:
    try:
        with open(path, 'rb') as fp:
            b = fp.read()
        sig, sr, audit = AudioPreprocessor.decode_wav_bytes(b)
        if sig is not None and audit.get("pass"):
            feats = AudioFeatureExtractor.extract_features(sig, sr)
            fv = feats["feature_vector"]
            with torch.inference_mode():
                t = torch.from_numpy(fv).float().unsqueeze(0)
                logits = pad(t)
                probs = torch.softmax(logits, dim=1).squeeze().tolist()
            res = voice_auth_engine.evaluate_anti_spoof(fv, feats["acoustic_metrics"])
            print(f"{path}: dur={audit.get('duration_sec')}s p_live={probs[0]:.4f} evaluated_liveness={res[0]} attack={res[2]} metrics={feats['acoustic_metrics']}")
    except Exception as e:
        print(f"Error on {path}: {e}")
