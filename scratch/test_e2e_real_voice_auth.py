import os
import sys
import io
import wave
import base64
import numpy as np
import torch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.voice_auth import voice_auth_engine, AudioPreprocessor

def main():
    print("==================================================")
    print("NIVORA END-TO-END REAL VOICE AUTHENTICATION AUDIT")
    print("==================================================")
    
    real_speech_dir = os.path.join(root_dir, "scratch", "real_speech")
    sample_files = sorted([os.path.join(real_speech_dir, f) for f in os.listdir(real_speech_dir) if f.endswith(".wav")])
    print(f"Total authentic dataset samples loaded: {len(sample_files)}")

    samples = []
    for sf in sample_files:
        with open(sf, "rb") as f:
            samples.append(f.read())
    
    # 1. Pipeline execution on authentic samples
    print("\n--- 1. Verification of Raw Authentic Speech Pipeline ---")
    pad_scores = []
    latencies = []
    for i, s_bytes in enumerate(samples[:4]):
        res = voice_auth_engine.process_voice_pipeline(s_bytes)
        pad_scores.append(res['liveness_score'])
        latencies.append(res['timings']['total_latency_ms'])
        print(f"Sample {i+1}: PAD Score={res['liveness_score']:.4f} ({res['liveness_verdict']}), "
              f"Latency={res['timings']['total_latency_ms']:.2f}ms "
              f"(ECAPA={res['timings']['ecapa_inference_ms']:.2f}ms, PAD={res['timings']['pad_inference_ms']:.2f}ms)")
        assert res['liveness_verdict'] == 'LIVE', f"Expected LIVE, got {res['liveness_verdict']}"
        assert res['liveness_score'] >= 0.70, f"Expected PAD >= 0.70, got {res['liveness_score']}"

    # 2. Test Speaker Verification: Enroll Speaker with samples 1, 2, 3
    print("\n--- 2. Multi-Sample Enrollment & Template Aggregation ---")
    res1 = voice_auth_engine.process_voice_pipeline(samples[0])
    res2 = voice_auth_engine.process_voice_pipeline(samples[1])
    res3 = voice_auth_engine.process_voice_pipeline(samples[2])
    
    template = voice_auth_engine.aggregate_embeddings([
        res1['probe_embedding'],
        res2['probe_embedding'],
        res3['probe_embedding']
    ])
    print(f"Aggregated template dimension: {len(template)}, norm: {np.linalg.norm(template):.6f}")
    assert len(template) == 192

    # 3. Same-Speaker Authentication with Sample 1
    print("\n--- 3. Same Speaker Authentication ---")
    res_auth = voice_auth_engine.process_voice_pipeline(samples[0], enrolled_template=template)
    print(f"Same-speaker similarity: {res_auth['voice_similarity']:.4f} (Threshold=0.65)")
    print(f"Speaker verdict: {res_auth['speaker_verdict']}")
    print(f"Audio PAD score: {res_auth['liveness_score']:.4f} (Threshold=0.70)")
    print(f"Audio PAD verdict: {res_auth['liveness_verdict']}")
    print(f"Final verdict: {res_auth['final_verdict']}")
    print(f"Detailed component timings:")
    for k, v in res_auth['timings'].items():
        print(f"  - {k}: {v} ms")
    assert res_auth['speaker_verdict'] == 'MATCH'
    assert res_auth['liveness_verdict'] == 'LIVE'
    assert res_auth['final_verdict'] == 'AUTHENTICATED'

    # 4. Imposter / Cross-Speaker Authentication
    print("\n--- 4. Cross-Speaker / Imposter Protection ---")
    res_imposter = voice_auth_engine.process_voice_pipeline(samples[5], enrolled_template=template)
    print(f"Cross-speaker similarity: {res_imposter['voice_similarity']:.4f} (Threshold=0.65)")
    print(f"Cross-speaker verdict: {res_imposter['speaker_verdict']}")
    print(f"Final verdict: {res_imposter['final_verdict']}")
    
    # 5. Audio Spoof Attack Detection
    print("\n--- 5. Audio Presentation Attack Detection (Fake Audio) ---")
    fake_wav_path = os.path.join(root_dir, "datasets", "voice_fake", "asvspoof_fake_01.wav")
    with open(fake_wav_path, "rb") as f:
        fake_bytes = f.read()
    res_fake = voice_auth_engine.process_voice_pipeline(fake_bytes, enrolled_template=template)
    print(f"Fake audio PAD score: {res_fake['liveness_score']:.4f} (Threshold=0.70)")
    print(f"Fake audio PAD verdict: {res_fake['liveness_verdict']}")
    print(f"Attack type detected: {res_fake.get('attack_type')}")
    print(f"Final verdict: {res_fake['final_verdict']}")
    assert res_fake['liveness_verdict'] == 'SPOOF'
    assert res_fake['final_verdict'] == 'REJECTED'

    print("\n==================================================")
    print("ALL MODULE 4 REAL-DATA AUDIT ASSERTIONS SUCCEEDED!")
    print("==================================================")

if __name__ == "__main__":
    main()
