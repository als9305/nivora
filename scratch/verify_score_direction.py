import sys
import os
import pandas as pd
import huggingface_hub

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.voice_auth import voice_auth_engine, AudioPreprocessor

p17 = huggingface_hub.hf_hub_download(
    repo_id="DynamicSuperb/SpoofDetection_ASVspoof2017",
    filename="data/test-00000-of-00001-e7ed034c56ededdd.parquet",
    repo_type="dataset"
)
df = pd.read_parquet(p17)

auth_bytes = df[df['label'] == 'authentic'].iloc[0]['audio']['bytes']
spoof_bytes = df[df['label'] == 'spoofed'].iloc[0]['audio']['bytes']

print("Processing AUTHENTIC (Bona Fide) sample:")
res_auth = voice_auth_engine.process_voice_pipeline(auth_bytes)
print("  Liveness score:", res_auth['liveness_score'])
print("  Voice spoof prob:", res_auth['voice_spoof_probability'])

print("\nProcessing SPOOFED (Replay) sample:")
res_spoof = voice_auth_engine.process_voice_pipeline(spoof_bytes)
print("  Liveness score:", res_spoof['liveness_score'])
print("  Voice spoof prob:", res_spoof['voice_spoof_probability'])
