import sys, os, numpy as np
sys.path.insert(0, ".")
from scripts.benchmark_voice_auth import fetch_librispeech_samples
from models.voice_auth import AudioPreprocessor, AudioFeatureExtractor, voice_auth_engine

libri = fetch_librispeech_samples()
print("Testing all 26 LibriSpeech speakers:")
passed = 0
total = 0
scores = []
for spk, utts in sorted(libri.items()):
    res = voice_auth_engine.process_voice_pipeline(utts[0])
    p = res["liveness_score"]
    status = "PASS" if p >= 0.70 else "FAIL"
    if p >= 0.70:
        passed += 1
    total += 1
    scores.append(p)
    print(f"Speaker {spk:<6}: score={p:.4f} verdict={res['liveness_verdict']} [{status}]")

print(f"\nTotal passed: {passed}/{total} ({passed/total*100:.1f}%)")
print(f"Min: {min(scores):.4f}, Max: {max(scores):.4f}, Median: {np.median(scores):.4f}")
