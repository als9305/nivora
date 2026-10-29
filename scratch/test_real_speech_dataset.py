import os
import sys
import numpy as np
import scipy.io.wavfile
import torch
from huggingface_hub import hf_hub_download

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.voice_auth import voice_auth_engine, AudioPreprocessor, AudioFeatureExtractor

def test_real_speech_files():
    print("=" * 65)
    print("TESTING REAL HUMAN SPEECH SAMPLES FROM SPEECHBRAIN (VOXCELEB/IEMOCAP)")
    print("=" * 65)

    # List of real human speech files from official SpeechBrain model repositories
    real_speech_sources = [
        ("VoxCeleb_Speaker1_Utt1", "speechbrain/spkrec-ecapa-voxceleb", "example1.wav"),
        ("VoxCeleb_Speaker2_Utt1", "speechbrain/spkrec-ecapa-voxceleb", "example2.flac"),
        ("VoxCeleb_Speaker3_Utt1", "speechbrain/spkrec-resnet-voxceleb", "example1.wav"),
        ("IEMOCAP_Speaker4_Anger", "speechbrain/emotion-recognition-wav2vec2-IEMOCAP", "anger.wav"),
        ("IEMOCAP_Speaker4_Happy", "speechbrain/emotion-recognition-wav2vec2-IEMOCAP", "hap.wav"),
        ("IEMOCAP_Speaker4_Neutral", "speechbrain/emotion-recognition-wav2vec2-IEMOCAP", "neutral.wav"),
        ("IEMOCAP_Speaker4_Sad", "speechbrain/emotion-recognition-wav2vec2-IEMOCAP", "sad.wav"),
        ("LibriSpeech_Speaker5_Utt1", "speechbrain/asr-crdnn-rnnlm-librispeech", "example.wav"),
        ("CommonVoice_Speaker6_Utt1", "speechbrain/asr-wav2vec2-commonvoice-en", "example.wav"),
    ]

    downloaded_samples = {}
    for name, repo, filename in real_speech_sources:
        try:
            p = hf_hub_download(repo_id=repo, filename=filename)
            with open(p, "rb") as f:
                raw_bytes = f.read()
            downloaded_samples[name] = {
                "path": p,
                "bytes": raw_bytes,
                "repo": repo,
                "filename": filename
            }
            print(f"Loaded {name:30s} from {repo:45s} ({len(raw_bytes)} bytes)")
        except Exception as e:
            print(f"Failed to load {name}: {e}")

    print("\nPROCESSED SAMPLES AUDIT:")
    decoded_embeddings = {}
    for name, data in downloaded_samples.items():
        sig, sr, audit = AudioPreprocessor.decode_wav_bytes(data["bytes"])
        if sig is None:
            print(f"  {name:30s} Preprocessing Failed: {audit.get('error_code')}")
            continue

        feats = AudioFeatureExtractor.extract_features(sig, sr)
        log_mel_80 = feats["log_mel_80"]
        # Apply CMVN
        log_mel_80_cmvn = log_mel_80 - np.mean(log_mel_80, axis=1, keepdims=True)
        emb = voice_auth_engine.extract_embedding(log_mel_80_cmvn)
        decoded_embeddings[name] = emb
        print(f"  {name:30s} | SR: {sr:5d}Hz | Dur: {audit['duration_sec']:5.2f}s | Emb Norm: {np.linalg.norm(emb):.4f}")

    print("\nCOSINE SIMILARITY COMPARISONS (REAL HUMAN SPEECH):")

    # Same Speaker Trials (IEMOCAP Speaker 4: Anger vs Happy vs Neutral vs Sad)
    if "IEMOCAP_Speaker4_Neutral" in decoded_embeddings and "IEMOCAP_Speaker4_Happy" in decoded_embeddings:
        sim_same_1 = voice_auth_engine.compute_similarity(decoded_embeddings["IEMOCAP_Speaker4_Neutral"], decoded_embeddings["IEMOCAP_Speaker4_Happy"])
        sim_same_2 = voice_auth_engine.compute_similarity(decoded_embeddings["IEMOCAP_Speaker4_Neutral"], decoded_embeddings["IEMOCAP_Speaker4_Sad"])
        sim_same_3 = voice_auth_engine.compute_similarity(decoded_embeddings["IEMOCAP_Speaker4_Anger"], decoded_embeddings["IEMOCAP_Speaker4_Happy"])
        print(f"  SAME SPEAKER (IEMOCAP Spk4 Neutral vs Happy):  {sim_same_1:.4f}")
        print(f"  SAME SPEAKER (IEMOCAP Spk4 Neutral vs Sad):    {sim_same_2:.4f}")
        print(f"  SAME SPEAKER (IEMOCAP Spk4 Anger vs Happy):    {sim_same_3:.4f}")

    # Impostor / Cross-Speaker Trials
    if "IEMOCAP_Speaker4_Neutral" in decoded_embeddings and "VoxCeleb_Speaker1_Utt1" in decoded_embeddings:
        sim_diff_1 = voice_auth_engine.compute_similarity(decoded_embeddings["IEMOCAP_Speaker4_Neutral"], decoded_embeddings["VoxCeleb_Speaker1_Utt1"])
        sim_diff_2 = voice_auth_engine.compute_similarity(decoded_embeddings["IEMOCAP_Speaker4_Neutral"], decoded_embeddings["LibriSpeech_Speaker5_Utt1"])
        sim_diff_3 = voice_auth_engine.compute_similarity(decoded_embeddings["VoxCeleb_Speaker1_Utt1"], decoded_embeddings["CommonVoice_Speaker6_Utt1"])
        print(f"  IMPOSTOR (IEMOCAP Spk4 vs VoxCeleb Spk1):      {sim_diff_1:.4f}")
        print(f"  IMPOSTOR (IEMOCAP Spk4 vs LibriSpeech Spk5):   {sim_diff_2:.4f}")
        print(f"  IMPOSTOR (VoxCeleb Spk1 vs CommonVoice Spk6):  {sim_diff_3:.4f}")

if __name__ == "__main__":
    test_real_speech_files()
