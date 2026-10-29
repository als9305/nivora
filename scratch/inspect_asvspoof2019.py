import huggingface_hub

api = huggingface_hub.HfApi()
print("Listing files in SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA...")
try:
    files = list(api.list_repo_files(repo_id="SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA", repo_type="dataset"))
    print("Files:", files[:15])
except Exception as e:
    print("Error:", e)

print("Listing files in LanceaKing/asvspoof2019...")
try:
    files2 = list(api.list_repo_files(repo_id="LanceaKing/asvspoof2019", repo_type="dataset"))
    print("Files:", files2[:15])
except Exception as e:
    print("Error:", e)
