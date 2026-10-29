import huggingface_hub

api = huggingface_hub.HfApi()
print("Searching HuggingFace datasets for 'asvspoof'...")
datasets = list(api.list_datasets(search="asvspoof", limit=10))
for d in datasets:
    print("Dataset ID:", d.id)
    try:
        files = list(api.list_repo_files(repo_id=d.id, repo_type="dataset"))
        print("  Files:", files[:8])
    except Exception as e:
        print("  Error:", e)
