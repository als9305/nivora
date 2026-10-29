import safetensors.torch
import huggingface_hub

p = huggingface_hub.hf_hub_download(repo_id="lab260/AASIST3", filename="model.safetensors")
tensors = safetensors.torch.load_file(p)
print("Keys in lab260/AASIST3 safetensors:")
for k in list(tensors.keys())[:20]:
    print(f"  {k}: {tensors[k].shape}")
print(f"Total keys: {len(tensors)}")
