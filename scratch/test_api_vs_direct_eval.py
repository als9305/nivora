import os
import sys
import base64
import json
import urllib.request
import numpy as np
import torch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.voice_auth import voice_auth_engine, AudioPreprocessor, AudioFeatureExtractor

print("=" * 80)
print("EVALUATING PRODUCTION API VS DIRECT MODEL INVOCATION")
print("=" * 80)

# Check API diagnostic
req = urllib.request.Request("http://127.0.0.1:8000/api/v1/voice/diagnostic")
with urllib.request.urlopen(req) as resp:
    diag = json.loads(resp.read().decode())
print("Live API Diagnostic:", diag)

# Enroll a dedicated test user on the live API with 3 genuine samples
enroll_files = [f"real_{i}.wav" for i in [1, 2, 3]]
enroll_b64 = []
for ef in enroll_files:
    p = os.path.join(root_dir, "scratch", "real_speech", ef)
    with open(p, "rb") as f:
        enroll_b64.append(base64.b64encode(f.read()).decode('ascii'))

enroll_payload = json.dumps({
    "user_id": "eval_test_user",
    "audio_samples_base64": enroll_b64
}).encode()

enroll_req = urllib.request.Request(
    "http://127.0.0.1:8000/api/v1/voice/enroll",
    data=enroll_payload,
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(enroll_req) as resp:
    enroll_res = json.loads(resp.read().decode())
print(f"Enrollment of 'eval_test_user': {enroll_res.get('status')}, template_id={enroll_res.get('template_id')}")

# Now collect all test files
test_files = []
# real speech
for i in range(1, 9):
    p = os.path.join(root_dir, "scratch", "real_speech", f"real_{i}.wav")
    if os.path.exists(p):
        test_files.append(("real_speech", f"real_{i}.wav", p, "GENUINE"))

# spoof
for name in sorted(os.listdir(os.path.join(root_dir, "datasets", "voice_fake"))):
    if name.endswith(".wav"):
        p = os.path.join(root_dir, "datasets", "voice_fake", name)
        test_files.append(("voice_fake", name, p, "SPOOF"))

print(f"\nCollected {len(test_files)} dataset test files.")

results = []
discrepancies = 0

for category, name, path, label in test_files:
    with open(path, "rb") as f:
        wav_bytes = f.read()

    # 1. Direct Engine Call (unconditioned on template for PAD evaluation)
    direct_res = voice_auth_engine.process_voice_pipeline(wav_bytes)
    direct_score = direct_res.get("liveness_score", 0.0)
    direct_spoof_prob = direct_res.get("voice_spoof_probability", 1.0)
    direct_verdict = direct_res.get("liveness_verdict", "REJECTED")

    # 2. Live API Call to /api/v1/voice/verify
    b64_audio = base64.b64encode(wav_bytes).decode('ascii')
    api_payload = json.dumps({
        "user_id": "eval_test_user",
        "audio_base64": b64_audio
    }).encode()
    api_req = urllib.request.Request(
        "http://127.0.0.1:8000/api/v1/voice/verify",
        data=api_payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(api_req) as resp:
        api_res = json.loads(resp.read().decode())
    
    api_score = api_res.get("liveness_score", 0.0)
    api_spoof_prob = api_res.get("voice_spoof_probability", 1.0)
    api_verdict = api_res.get("liveness_verdict", "REJECTED")
    api_speaker_sim = api_res.get("voice_similarity", 0.0)
    api_speaker_verdict = api_res.get("speaker_verdict", "MISMATCH")
    api_final_verdict = api_res.get("final_verdict", "REJECTED")

    # Compare direct vs API
    score_diff = abs(direct_score - api_score)
    if score_diff > 0.001:
        discrepancies += 1
        print(f"DISCREPANCY on {name}: direct_score={direct_score} vs api_score={api_score}")

    results.append({
        "category": category,
        "name": name,
        "label": label,
        "direct_score": direct_score,
        "api_score": api_score,
        "direct_spoof_prob": direct_spoof_prob,
        "api_spoof_prob": api_spoof_prob,
        "direct_verdict": direct_verdict,
        "api_verdict": api_verdict,
        "speaker_sim": api_speaker_sim,
        "speaker_verdict": api_speaker_verdict,
        "final_verdict": api_final_verdict,
        "pass_pad": direct_score >= 0.70
    })

print(f"\nComparison Complete. Total Discrepancies between API and Direct Engine: {discrepancies}")

print("\n--- SUMMARY OF EVALUATION (PAD Threshold = 0.70, Speaker Threshold = 0.65) ---")
print(f"{'Name':<24} {'Label':<8} {'Score':<8} {'SpoofProb':<10} {'PAD':<8} {'SpkSim':<8} {'SpkVerd':<9} {'Final':<13} {'Result':<12}")
print("-" * 105)

genuine_total = 0
genuine_passed = 0
spoof_total = 0
spoof_rejected = 0

for r in results:
    is_genuine = (r["label"] == "GENUINE")
    if is_genuine:
        genuine_total += 1
        if r["pass_pad"]:
            genuine_passed += 1
            res_str = "CORRECT"
        else:
            res_str = "FALSE_REJECT"
    else:
        spoof_total += 1
        if not r["pass_pad"]:
            spoof_rejected += 1
            res_str = "CORRECT"
        else:
            res_str = "FALSE_ACCEPT"

    print(f"{r['name']:<24} {r['label']:<8} {r['api_score']:<8.4f} {r['api_spoof_prob']:<10.4f} {r['api_verdict']:<8} {r['speaker_sim']:<8.4f} {r['speaker_verdict']:<9} {r['final_verdict']:<13} {res_str:<12}")

print("-" * 105)
print(f"GENUINE (Bona-Fide): {genuine_passed}/{genuine_total} passed ({genuine_passed/genuine_total*100:.1f}%), False Rejections: {genuine_total - genuine_passed}")
print(f"SPOOF: {spoof_rejected}/{spoof_total} rejected ({spoof_rejected/spoof_total*100:.1f}%), False Acceptances: {spoof_total - spoof_rejected}")
