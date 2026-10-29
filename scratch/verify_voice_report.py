import os
import sys
import glob
import time
import pandas as pd
import numpy as np
import pathlib

sys.path.insert(0, os.path.abspath("."))
from models.voice_auth import voice_auth_engine, AudioPreprocessor, AudioFeatureExtractor

def run_acceptance_verification():
    print("=" * 70)
    print("NIVORA MODULE 4 VOICE AUTHENTICATION & PAD ACCEPTANCE VERIFICATION")
    print("=" * 70)

    # 1. Check Model Loading & Pretrained Status
    spk_pretrained = voice_auth_engine.embedding_net.is_pretrained
    spk_source = voice_auth_engine.embedding_net.model_info["weight_source"]
    pad_pretrained = voice_auth_engine.anti_spoof_net.is_pretrained
    pad_source = voice_auth_engine.anti_spoof_net.model_info["weight_source"]

    print(f"\n[SPEAKER MODEL] ECAPA-TDNN Pretrained: {spk_pretrained}")
    print(f"                Source: {spk_source}")
    print(f"                Embedding Dimension: {voice_auth_engine.embedding_net.embedding_dim}-D")
    print(f"                Speaker Threshold: {voice_auth_engine.tau_speaker}")

    print(f"\n[AUDIO PAD MODEL] PassiveAudioAntiSpoofNet Pretrained: {pad_pretrained}")
    print(f"                  Source: {pad_source}")
    print(f"                  Input Features: 64-D Acoustic Descriptors")
    print(f"                  PAD Threshold: {voice_auth_engine.tau_liveness}")

    # 2. Load Authentic Speech from cached dataset
    cache_path = list(pathlib.Path(os.path.expanduser("~/.cache/huggingface")).rglob("*e7ed034c56ededdd.parquet"))
    if cache_path:
        df = pd.read_parquet(cache_path[0])
        auth_rows = df[df["label"] == "authentic"].head(5)
    else:
        auth_rows = []

    print(f"\nLoaded {len(auth_rows)} authentic speech samples from benchmark cache.")

    # Enroll user with first authentic sample
    enroll_audio_b = auth_rows.iloc[0]["audio"]["bytes"]
    enroll_sig, enroll_sr, _ = AudioPreprocessor.decode_wav_bytes(enroll_audio_b)
    enroll_feats = AudioFeatureExtractor.extract_features(enroll_sig, enroll_sr)
    enroll_emb = voice_auth_engine.extract_embedding(enroll_feats["log_mel_80"])
    print(f"[ENROLLMENT] Enrolled genuine template (shape: {len(enroll_emb)}-D, L2 norm: {np.linalg.norm(enroll_emb):.4f})")

    # Evaluate authentic samples
    print("\n" + "=" * 70)
    print("1. EVALUATING BONA FIDE GENUINE SPEECH SAMPLES (EXPECTED: LIVE, >= 0.70)")
    print("=" * 70)

    for i in range(len(auth_rows)):
        audio_b = auth_rows.iloc[i]["audio"]["bytes"]
        t0 = time.perf_counter()
        res = voice_auth_engine.process_voice_pipeline(audio_b, enrolled_template=enroll_emb)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        score = res["liveness_score"]
        sim = res["voice_similarity"]
        l_verdict = res["liveness_verdict"]
        s_verdict = res["speaker_verdict"]
        final_v = res["final_verdict"]

        print(f"Genuine Sample {i+1}:")
        print(f"  PAD Score: {score:.4f} (Threshold: 0.70) -> {l_verdict}")
        print(f"  Speaker Similarity: {sim:.4f} (Threshold: 0.65) -> {s_verdict}")
        print(f"  Final Verdict: {final_v} | Latency: {latency_ms:.1f}ms")

    # 3. Evaluate Fake / Presentation Attack samples
    fake_files = sorted(glob.glob("datasets/voice_fake/*.wav"))
    print("\n" + "=" * 70)
    print(f"2. EVALUATING PRESENTATION ATTACKS / FAKE SPEECH ({len(fake_files)} SAMPLES, EXPECTED: SPOOF, < 0.70)")
    print("=" * 70)

    fake_scores = []
    for ff in fake_files:
        with open(ff, "rb") as f:
            b = f.read()
        t0 = time.perf_counter()
        res = voice_auth_engine.process_voice_pipeline(b, enrolled_template=enroll_emb)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        score = res["liveness_score"]
        l_verdict = res["liveness_verdict"]
        s_verdict = res["speaker_verdict"]
        final_v = res["final_verdict"]
        attack = res.get("attack_type")
        fake_scores.append(score)

        print(f"[{os.path.basename(ff)}] PAD Score: {score:.4f} -> {l_verdict} (Attack: {attack}) | Final: {final_v} | Latency: {latency_ms:.1f}ms")

    all_rejected = all(s < 0.70 for s in fake_scores)
    max_fake = max(fake_scores) if fake_scores else 0.0
    print(f"\nAttack Detection Summary:")
    print(f"  Total Fake Files: {len(fake_files)}")
    print(f"  Max Fake PAD Score: {max_fake:.4f} (Threshold: 0.70)")
    print(f"  100% Spoof Rejection Rate: {all_rejected}")

if __name__ == "__main__":
    run_acceptance_verification()
