import os
import sys
import numpy as np
import torch
import huggingface_hub
import pandas as pd
import io, wave

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.voice_auth import voice_auth_engine, AudioPreprocessor, AudioFeatureExtractor

def vad_trim(signal: np.ndarray, framerate: int = 16000) -> np.ndarray:
    """Trims leading and trailing silence using energy-based VAD with 100ms margin."""
    frame_len = int(framerate * 0.02)
    hop_len = int(framerate * 0.01)
    if len(signal) < frame_len * 2:
        return signal
    
    num_frames = (len(signal) - frame_len) // hop_len + 1
    energies = np.array([
        np.sum(signal[i * hop_len : i * hop_len + frame_len] ** 2)
        for i in range(num_frames)
    ])
    
    # Adaptive threshold: 30th percentile * 2.5, clamped
    thresh = max(0.001, np.percentile(energies, 30) * 2.5)
    speech_indices = np.where(energies > thresh)[0]
    
    if len(speech_indices) == 0:
        return signal
    
    # Include a safety margin of ~150ms before and after speech
    margin_frames = int(0.15 * framerate / hop_len)
    start_frame = max(0, speech_indices[0] - margin_frames)
    end_frame = min(num_frames - 1, speech_indices[-1] + margin_frames)
    
    start_sample = start_frame * hop_len
    end_sample = min(len(signal), (end_frame + 1) * hop_len + frame_len)
    
    trimmed = signal[start_sample:end_sample]
    return trimmed if len(trimmed) >= int(framerate * 0.3) else signal

p17 = huggingface_hub.hf_hub_download(
    repo_id="DynamicSuperb/SpoofDetection_ASVspoof2017",
    filename="data/test-00000-of-00001-e7ed034c56ededdd.parquet",
    repo_type="dataset"
)
df17 = pd.read_parquet(p17)

print("=== TESTING VAD TRIMMING ON NOISY/PADDED SAMPLES ===")
for sample_idx in range(5):
    auth_row = df17[df17['label'] == 'authentic'].iloc[sample_idx]
    audio_b = auth_row['audio']['bytes']
    sig, sr, _ = AudioPreprocessor.decode_wav_bytes(audio_b)
    
    # Add 1.0s room noise before and after
    noise = np.random.normal(0, 0.005, sr).astype(np.float32)
    sig_noisy = np.concatenate([noise, sig, noise])
    
    # Without VAD trimming:
    feats_raw = AudioFeatureExtractor.extract_features(sig_noisy, sr)
    score_raw, _, _ = voice_auth_engine.evaluate_anti_spoof(feats_raw['feature_vector'], feats_raw['acoustic_metrics'])
    
    # With VAD trimming:
    sig_trimmed = vad_trim(sig_noisy, sr)
    feats_trimmed = AudioFeatureExtractor.extract_features(sig_trimmed, sr)
    score_trimmed, _, _ = voice_auth_engine.evaluate_anti_spoof(feats_trimmed['feature_vector'], feats_trimmed['acoustic_metrics'])
    
    print(f"Sample {sample_idx}:")
    print(f"  Raw with noise:     score={score_raw:.4f}")
    print(f"  VAD trimmed noise:  score={score_trimmed:.4f}")
