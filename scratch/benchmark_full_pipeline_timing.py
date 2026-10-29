import os
import sys
import time
import base64
import cv2
import numpy as np
import asyncio
import torch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(root_dir, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.services.face_service import FaceService
from app.services.liveness_service import LivenessService
from app.services.deepfake_service import DeepfakeService
from app.services.voice_service import VoiceService, VoiceSessionManager
from app.services.multimodal_fusion_service import MultimodalFusionService, get_deepfake_analyzer
from app.schemas.multimodal import MultimodalAuthRequest
from models.voice_auth import voice_auth_engine

async def run_benchmark():
    print("=" * 80)
    print("BASELINE COMPREHENSIVE PIPELINE LATENCY PROFILING")
    print("=" * 80)

    # 1. Prepare realistic test assets
    # Face probe image: 640x480 realistic face canvas
    face_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(face_img, (0, 0), (640, 480), (70, 70, 70), -1)
    # Head ellipse
    cv2.ellipse(face_img, (320, 240), (100, 140), 0, 0, 360, (190, 170, 150), -1)
    # Eyes
    cv2.circle(face_img, (280, 210), 12, (255, 255, 255), -1)
    cv2.circle(face_img, (360, 210), 12, (255, 255, 255), -1)
    cv2.circle(face_img, (280, 210), 5, (20, 20, 20), -1)
    cv2.circle(face_img, (360, 210), 5, (20, 20, 20), -1)
    # Nose & mouth
    cv2.line(face_img, (320, 220), (320, 260), (140, 120, 100), 3)
    cv2.ellipse(face_img, (320, 290), (35, 15), 0, 0, 180, (80, 50, 160), -1)

    probe_b64 = "data:image/jpeg;base64," + base64.b64encode(cv2.imencode('.jpg', face_img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode('utf-8')

    # Sequence of 18 frames for liveness & deepfake (slightly varied to simulate movement)
    seq_b64 = []
    for i in range(18):
        f = face_img.copy()
        # slight offset
        dx = int((i - 9) * 3)
        cv2.ellipse(f, (320 + dx, 240), (100, 140), 0, 0, 360, (190, 170, 150), -1)
        b64 = "data:image/jpeg;base64," + base64.b64encode(cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode('utf-8')
        seq_b64.append(b64)

    # Voice probe: real_1.wav
    voice_path = os.path.join(root_dir, "scratch", "real_speech", "real_1.wav")
    with open(voice_path, "rb") as fp:
        wav_bytes = fp.read()
    audio_b64 = "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode('utf-8')

    user_id = "latency_prof_user"

    # Enroll user for face and voice
    print("[Setup] Enrolling user for realistic end-to-end testing...")
    await FaceService.enroll_face(user_id, [probe_b64, seq_b64[0], seq_b64[1]])
    await VoiceService.enroll_voice(user_id, [audio_b64, audio_b64, audio_b64])

    print("\n--- MEASURING INDIVIDUAL STAGES (Cold vs Warm) ---")

    # Module 1: DeepQShield (18 frames)
    t0 = time.perf_counter()
    res_m1_cold = await DeepfakeService.analyze_deepfake(user_id, seq_b64[:18])
    t_m1_cold = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    res_m1_warm = await DeepfakeService.analyze_deepfake(user_id, seq_b64[:18])
    t_m1_warm = (time.perf_counter() - t0) * 1000
    print(f"Module 1 (DeepQShield, 18 frames): Cold = {t_m1_cold:.2f} ms | Warm = {t_m1_warm:.2f} ms")

    # Module 2: Face Identity (1 frame)
    t0 = time.perf_counter()
    res_m2_cold = await FaceService.verify_face(user_id, probe_b64)
    t_m2_cold = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    res_m2_warm = await FaceService.verify_face(user_id, probe_b64)
    t_m2_warm = (time.perf_counter() - t0) * 1000
    print(f"Module 2 (Face Identity, 1 frame): Cold = {t_m2_cold:.2f} ms | Warm = {t_m2_warm:.2f} ms")

    # Module 3: Face Liveness (18 frames)
    t0 = time.perf_counter()
    res_m3_cold = LivenessService.evaluate_passive_liveness(seq_b64[:18], challenge="TURN_LEFT")
    t_m3_cold = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    res_m3_warm = LivenessService.evaluate_passive_liveness(seq_b64[:18], challenge="TURN_LEFT")
    t_m3_warm = (time.perf_counter() - t0) * 1000
    print(f"Module 3 (Face Liveness, 18 frames): Cold = {t_m3_cold:.2f} ms | Warm = {t_m3_warm:.2f} ms")

    # Module 4: Voice Authentication
    t0 = time.perf_counter()
    sess_doc = VoiceSessionManager.create_session(user_id, "VERIFY")
    sess_id = sess_doc["session_id"]
    res_m4_cold = await VoiceService.verify_voice(user_id, audio_b64, session_id=sess_id)
    t_m4_cold = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    sess_doc2 = VoiceSessionManager.create_session(user_id, "VERIFY")
    sess_id2 = sess_doc2["session_id"]
    res_m4_warm = await VoiceService.verify_voice(user_id, audio_b64, session_id=sess_id2)
    t_m4_warm = (time.perf_counter() - t0) * 1000
    print(f"Module 4 (Voice Auth & PAD): Cold = {t_m4_cold:.2f} ms | Warm = {t_m4_warm:.2f} ms")

    # Module 5: Multimodal Fusion Orchestrator
    # First: without session token (direct orchestration)
    req_direct = MultimodalAuthRequest(
        user_id=user_id,
        face_image_base64=probe_b64,
        face_frames_base64=seq_b64[:18],
        audio_pcm_base64=audio_b64
    )
    t0 = time.perf_counter()
    res_m5_cold = await MultimodalFusionService.authenticate_multimodal(req_direct)
    t_m5_cold = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    res_m5_warm = await MultimodalFusionService.authenticate_multimodal(req_direct)
    t_m5_warm = (time.perf_counter() - t0) * 1000
    print(f"Module 5 (Unified Direct Orchestration): Cold = {t_m5_cold:.2f} ms | Warm = {t_m5_warm:.2f} ms")

    # Now simulate the frontend's exact 5-step sequential flow
    print("\n--- SIMULATING CLIENT 5-STEP SEQUENTIAL PIPELINE (OPTIMIZED SESSION-BOUND) ---")
    t_seq_start = time.perf_counter()
    # Pre-start session
    s_doc = VoiceSessionManager.create_session(user_id, "VERIFY")
    s_id = s_doc["session_id"]

    # Step 1: Face ID
    t0 = time.perf_counter()
    r1 = await FaceService.verify_face(user_id, probe_b64)
    VoiceSessionManager.store_face_evidence(s_id, user_id, r1)
    t_step1 = (time.perf_counter() - t0) * 1000

    # Step 2: Liveness
    t0 = time.perf_counter()
    r2 = LivenessService.evaluate_passive_liveness(seq_b64[:18], challenge="TURN_LEFT")
    VoiceSessionManager.store_liveness_evidence(s_id, user_id, r2)
    t_step2 = (time.perf_counter() - t0) * 1000

    # Step 3: Voice Start + Verify (runs ASR)
    t0 = time.perf_counter()
    r3 = await VoiceService.verify_voice(user_id, audio_b64, session_id=s_id)
    t_step3 = (time.perf_counter() - t0) * 1000

    # Step 4: DeepQShield
    t0 = time.perf_counter()
    r4 = await DeepfakeService.analyze_deepfake(user_id, seq_b64[:18])
    VoiceSessionManager.store_deepfake_evidence(s_id, user_id, r4)
    t_step4 = (time.perf_counter() - t0) * 1000

    # Step 5: Fusion with session challenge token
    t0 = time.perf_counter()
    fusion_req = MultimodalAuthRequest(
        user_id=user_id,
        challenge_token=s_id,
        face_image_base64=probe_b64,
        face_frames_base64=seq_b64[:18],
        audio_pcm_base64=""
    )
    r5 = await MultimodalFusionService.authenticate_multimodal(fusion_req)
    t_step5 = (time.perf_counter() - t0) * 1000

    total_seq_time = (time.perf_counter() - t_seq_start) * 1000
    print(f"  Step 1 (Face Identity):       {t_step1:.2f} ms")
    print(f"  Step 2 (Face Liveness):       {t_step2:.2f} ms")
    print(f"  Step 3 (Voice Authentication):{t_step3:.2f} ms (Phrase Match: {r3.get('phrase_match')})")
    print(f"  Step 4 (DeepQShield):         {t_step4:.2f} ms")
    print(f"  Step 5 (Final Fusion):        {t_step5:.2f} ms (Decision: {r5.decision})")
    print(f"  TOTAL CLIENT WORKFLOW TIME:   {total_seq_time:.2f} ms ({total_seq_time/1000:.2f} s)")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
