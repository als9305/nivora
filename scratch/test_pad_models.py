import os
import torch
import huggingface_hub

print("Testing HuggingFace Hub downloading for ASVspoof PAD models...")

models_to_try = [
    ("ALLA1N/rawnet2-la-clean-specialist", "pytorch_model.bin"),
    ("lab260/AASIST3", "model.safetensors"),
    ("lab260/AASIST3", "pytorch_model.bin"),
    ("ash56/ssl-aasist", "model.pth"),
    ("rpeddu/enhanced-rawnet2-antispoofing", "pytorch_model.bin"),
    ("SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA", "README.md")
]

for repo_id, filename in models_to_try:
    try:
        path = huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename)
        print(f"[SUCCESS] Downloaded {repo_id}/{filename} -> {path}")
    except Exception as e:
        print(f"[FAIL] {repo_id}/{filename}: {e}")
