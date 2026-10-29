import os
import torch
import huggingface_hub

repos = [
    ("caa-speech-detection-asvspoof2019/rawnet2", "best.pt"),
    ("codecfakev2/model_LA_WCE_4_14_1e-06_ASVSPOOF19_16k", "epoch_3.pth"),
    ("arnabdas8901/aasist-trained-asvspoof2024", "orig_aasist_epoch_1.pth")
]

for repo, fname in repos:
    try:
        p = huggingface_hub.hf_hub_download(repo_id=repo, filename=fname)
        print(f"[SUCCESS] Downloaded {repo}/{fname} -> {p}")
        ckpt = torch.load(p, map_location='cpu')
        if isinstance(ckpt, dict):
            print(f"  Keys in checkpoint: {list(ckpt.keys())[:10]}")
        else:
            print(f"  Loaded model/tensor type: {type(ckpt)}")
    except Exception as e:
        print(f"[FAIL] {repo}/{fname}: {e}")
