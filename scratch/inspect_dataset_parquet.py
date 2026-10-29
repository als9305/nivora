import pandas as pd
import huggingface_hub

print("Downloading parquet sample from DynamicSuperb/SpoofDetection_ASVspoof2017...")
path = huggingface_hub.hf_hub_download(
    repo_id="DynamicSuperb/SpoofDetection_ASVspoof2017",
    filename="data/test-00000-of-00001-e7ed034c56ededdd.parquet",
    repo_type="dataset"
)

df = pd.read_parquet(path)
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("Head:")
print(df.head(3))
if 'label' in df.columns:
    print("Label distribution:")
    print(df['label'].value_counts())
elif 'target' in df.columns:
    print("Target distribution:")
    print(df['target'].value_counts())
