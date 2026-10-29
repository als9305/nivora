import os
import sys
import torch
import numpy as np

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.voice_auth import voice_auth_engine, AudioPreprocessor, AudioFeatureExtractor

print("=== VOICE AUTH ENGINE DIAGNOSTICS ===")
print("Speaker model:", voice_auth_engine.embedding_net.model_info)
print("PAD model:", voice_auth_engine.anti_spoof_net.model_info)
print("PAD model is_pretrained:", voice_auth_engine.anti_spoof_net.is_pretrained)
print("tau_speaker:", voice_auth_engine.tau_speaker)
print("tau_liveness:", voice_auth_engine.tau_liveness)

# Test 1: Check the weights in trained_pad_model_v3.pt
ckpt_path = "models/trained_pad_model_v3.pt"
if os.path.exists(ckpt_path):
    st = torch.load(ckpt_path, map_location='cpu')
    print("Checkpoint keys:", st.keys())
    sd = st.get('model_state_dict', {})
    for k, v in sd.items():
        print(f"  {k}: shape {v.shape}, mean {v.mean().item():.4f}, std {v.std().item():.4f}")

# Test 2: Test evaluate_anti_spoof with zeros, random, and various acoustic metrics
for name, fvec in [
    ("Zeros", np.zeros(64, dtype=np.float32)),
    ("Ones", np.ones(64, dtype=np.float32)),
    ("Random Normal", np.random.randn(64).astype(np.float32)),
]:
    with torch.no_grad():
        t = torch.from_numpy(fvec).float().unsqueeze(0)
        logits = voice_auth_engine.anti_spoof_net(t)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
    print(f"\nFeature [{name}]:")
    print(f"  Logits: {logits.numpy().tolist()}")
    print(f"  Softmax (class 0 = BONAFIDE, class 1 = SPOOF): {probs.tolist()}")
    score, spoof, att = voice_auth_engine.evaluate_anti_spoof(fvec, {})
    print(f"  Final score: {score}, spoof: {spoof}, attack: {att}")
