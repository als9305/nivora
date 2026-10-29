import pandas as pd
import numpy as np
import huggingface_hub

print("Checking ASVspoof 2017 Parquet...")
p17 = huggingface_hub.hf_hub_download(repo_id="DynamicSuperb/SpoofDetection_ASVspoof2017", filename="data/test-00000-of-00001-e7ed034c56ededdd.parquet", repo_type="dataset")
df17 = pd.read_parquet(p17)
print("ASVspoof2017 shape:", df17.shape)
print("ASVspoof2017 labels:", df17['label'].value_counts().to_dict())

print("\nChecking ASVspoof 2015 Parquet...")
p15 = huggingface_hub.hf_hub_download(repo_id="DynamicSuperb/SpoofDetection_ASVspoof2015", filename="data/test-00000-of-00001-4a252b1e92dc86af.parquet", repo_type="dataset")
df15 = pd.read_parquet(p15)
print("ASVspoof2015 shape:", df15.shape)
print("ASVspoof2015 labels:", df15['label'].value_counts().to_dict())
