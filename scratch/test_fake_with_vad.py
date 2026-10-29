import os
import sys
import glob
import numpy as np
import torch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.voice_auth import voice_auth_engine, AudioPreprocessor, AudioFeatureExtractor
from scratch.test_vad_trim import vad_trim

print("=== TESTING FAKE SAMPLES WITH VAD TRIMMING ===")
for f in sorted(glob.glob("datasets/voice_fake/*.wav")):
    sig, sr, audit = AudioPreprocessor.decode_wav_bytes(open(f, 'rb').read())
    sig_trimmed = vad_trim(sig, sr)
    feats = AudioFeatureExtractor.extract_features(sig_trimmed, sr)
    score, spoof, att = voice_auth_engine.evaluate_anti_spoof(feats['feature_vector'], feats['acoustic_metrics'])
    print(f"Fake {os.path.basename(f)}: score={score:.4f}, spoof_prob={spoof:.4f}, verdict={'LIVE' if score>=0.70 else 'SPOOF'}, attack={att}")
