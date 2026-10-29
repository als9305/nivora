import os
import sys
import numpy as np
import huggingface_hub
import pandas as pd

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.voice_auth import voice_auth_engine, AudioPreprocessor, AudioFeatureExtractor
from scratch.test_vad_trim import vad_trim

p17 = huggingface_hub.hf_hub_download(
    repo_id="DynamicSuperb/SpoofDetection_ASVspoof2017",
    filename="data/test-00000-of-00001-e7ed034c56ededdd.parquet",
    repo_type="dataset"
)
df17 = pd.read_parquet(p17)

# Take 2 samples from authentic speakers
s1 = df17[df17['label'] == 'authentic'].iloc[1]['audio']['bytes']
sig1, sr, _ = AudioPreprocessor.decode_wav_bytes(s1)

# Add noise and silence to simulate 2 different utterances from same user
noise1 = np.random.normal(0, 0.005, sr).astype(np.float32)
sig1_noisy = np.concatenate([noise1, sig1, noise1])

feats_clean = AudioFeatureExtractor.extract_features(sig1, sr)
emb_clean = voice_auth_engine.extract_embedding(feats_clean['log_mel_80'])

# Without VAD trimming:
feats_noisy = AudioFeatureExtractor.extract_features(sig1_noisy, sr)
emb_noisy = voice_auth_engine.extract_embedding(feats_noisy['log_mel_80'])
sim_noisy = voice_auth_engine.compute_similarity(emb_clean, emb_noisy)

# With VAD trimming:
sig1_trimmed = vad_trim(sig1_noisy, sr)
feats_trimmed = AudioFeatureExtractor.extract_features(sig1_trimmed, sr)
emb_trimmed = voice_auth_engine.extract_embedding(feats_trimmed['log_mel_80'])
sim_trimmed = voice_auth_engine.compute_similarity(emb_clean, emb_trimmed)

print(f"Similarity between clean utterance and noisy utterance without VAD: {sim_noisy:.4f}")
print(f"Similarity between clean utterance and noisy utterance WITH VAD:    {sim_trimmed:.4f}")
