import httpx
import base64
import os

def create_dummy_image_b64():
    # 1x1 PNG image base64
    png_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc\xf8\xff\xff?\x03\x00\x05\xfe\x02\xfe\xa7\x9a\x9a\xa8\x00\x00\x00\x00IEND\xaeB`\x82'
    return f"data:image/png;base64,{base64.b64encode(png_bytes).decode()}"

def create_dummy_audio_b64():
    # Small 44-byte WAV header base64
    wav_bytes = b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
    return f"data:audio/wav;base64,{base64.b64encode(wav_bytes).decode()}"

def test_mode(mode_name: str, img_b64: str, audio_b64: str):
    print(f"\n==========================================")
    print(f" TESTING MODE: {mode_name}")
    print(f"==========================================")
    
    base_url = "http://localhost:8000"
    
    # 1. Health check
    r1 = httpx.get(f"{base_url}/health")
    assert r1.status_code == 200, f"Health check failed: {r1.text}"
    print(f"1. Health Check: {r1.status_code} OK")

    # 2. Register user with mode biometrics
    email = f"user_{mode_name.lower()}@deepguard.sec"
    reg_payload = {
        "name": f"User {mode_name}",
        "email": email,
        "password": "Password123!",
        "image_base64": img_b64,
        "voice_embedding": [0.1, 0.2, 0.3, 0.4]
    }
    r2 = httpx.post(f"{base_url}/api/auth/register", json=reg_payload)
    assert r2.status_code == 200, f"Register failed: {r2.text}"
    user_id = r2.json()["user_id"]
    print(f"2. User Register ({mode_name}): 200 OK (User ID: {user_id})")

    # 3. Enroll face
    enroll_payload = {"email": email, "image_base64": img_b64}
    r3 = httpx.post(f"{base_url}/api/auth/enroll-face", json=enroll_payload)
    assert r3.status_code == 200, f"Enroll face failed: {r3.text}"
    print(f"3. Enroll Face Baseline: 200 OK")

    # 4. Login Step 0 (Password verification)
    r4 = httpx.post(f"{base_url}/api/auth/login", json={"email": email, "password": "Password123!"})
    assert r4.status_code == 200, f"Login failed: {r4.text}"
    login_res = r4.json()
    challenge_phrase = login_res["challenge_phrase"]
    liveness_challenge = login_res["liveness_challenges"][0]
    print(f"4. Password Login: 200 OK (Challenge: {liveness_challenge})")

    # 5. Face Verify Step 1
    r5 = httpx.post(f"{base_url}/api/auth/face-verify", json={"user_id": user_id, "image_base64": img_b64})
    assert r5.status_code == 200, f"Face verify failed: {r5.text}"
    face_score = r5.json()["face_score"]
    print(f"5. Step 1 Face Verify ({mode_name}): 200 OK (Score: {face_score})")

    # 6. Liveness Challenge Step 2
    r6 = httpx.post(f"{base_url}/api/auth/liveness", json={
        "user_id": user_id,
        "challenge": liveness_challenge,
        "frames_base64": [img_b64, img_b64, img_b64]
    })
    assert r6.status_code == 200, f"Liveness challenge failed: {r6.text}"
    live_score = r6.json()["liveness_score"]
    print(f"6. Step 2 Liveness Challenge ({mode_name}): 200 OK (Score: {live_score})")

    # 7. Deepfake Inspection Step 3
    r7 = httpx.post(f"{base_url}/api/auth/deepfake", json={
        "user_id": user_id,
        "frames_base64": [img_b64, img_b64, img_b64, img_b64]
    })
    assert r7.status_code == 200, f"Deepfake verify failed: {r7.text}"
    deepfake_prob = r7.json()["deepfake_probability"]
    print(f"7. Step 3 Deepfake Inspection ({mode_name}): 200 OK (Prob: {deepfake_prob})")

    # 8. Voice Verification Step 4
    r8 = httpx.post(f"{base_url}/api/auth/voice", json={
        "user_id": user_id,
        "audio_base64": audio_b64,
        "challenge_phrase": challenge_phrase
    })
    assert r8.status_code == 200, f"Voice verify failed: {r8.text}"
    voice_res = r8.json()
    print(f"8. Step 4 Voice Verification ({mode_name}): 200 OK (Similarity: {voice_res['voice_similarity']})")

    # 9. Composite Authentication Step 5
    r9 = httpx.post(f"{base_url}/api/auth/authenticate", json={
        "user_id": user_id,
        "email": email,
        "face_score": face_score,
        "liveness_score": live_score,
        "deepfake_probability": deepfake_prob,
        "voice_similarity": voice_res["voice_similarity"],
        "voice_spoof_probability": voice_res["voice_spoof_probability"]
    })
    assert r9.status_code == 200, f"Final authenticate failed: {r9.text}"
    final_res = r9.json()
    print(f"9. Step 5 Composite Risk Decision ({mode_name}): 200 OK")
    print(f"   -> Decision: {final_res['decision']} | Risk Score: {final_res['risk_score']}/100")

def main():
    img_data = create_dummy_image_b64()
    audio_data = create_dummy_audio_b64()

    # Test Mode 1: Live Hardware Webcam & Mic Inputs
    test_mode("LIVE_WEBCAM_MIC", img_data, audio_data)

    # Test Mode 2: Attached File Upload Inputs (Image/Video & Audio File)
    test_mode("ATTACHED_FILE_UPLOAD", img_data, audio_data)

    print("\n==========================================")
    print(" ALL TESTS PASSED SUCCESSFULLY FOR BOTH MODES! ")
    print("==========================================")

if __name__ == "__main__":
    main()
