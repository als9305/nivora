import torch
import huggingface_hub

for model_name in ["rawnet2", "lcnn"]:
    repo_id = f"caa-speech-detection-asvspoof2019/{model_name}"
    try:
        p = huggingface_hub.hf_hub_download(repo_id=repo_id, filename="best.pt")
        ckpt = torch.load(p, map_location='cpu')
        print(f"\n[SUCCESS] {repo_id}:")
        if isinstance(ckpt, dict):
            print("  Keys:", list(ckpt.keys()))
            if 'state_dict' in ckpt:
                sd = ckpt['state_dict']
                print("  State dict keys count:", len(sd))
                print("  First 5 SD keys:", list(sd.keys())[:5])
            elif 'model_state_dict' in ckpt:
                sd = ckpt['model_state_dict']
                print("  Model state dict keys count:", len(sd))
                print("  First 5 SD keys:", list(sd.keys())[:5])
        else:
            print("  Loaded object type:", type(ckpt))
    except Exception as e:
        print(f"[FAIL] {repo_id}: {e}")
