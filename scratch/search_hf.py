import huggingface_hub

api = huggingface_hub.HfApi()
print("Searching HuggingFace models for 'asvspoof'...")
models = list(api.list_models(search="asvspoof", limit=20))
for m in models:
    print("Model ID:", m.id)
    try:
        files = list(api.list_repo_files(repo_id=m.id))
        print("  Files:", files[:5])
    except Exception as e:
        print("  Could not list files:", e)

print("\nSearching HuggingFace models for 'audio anti-spoofing'...")
models_pad = list(api.list_models(search="anti-spoofing", limit=20))
for m in models_pad:
    print("Model ID:", m.id)
