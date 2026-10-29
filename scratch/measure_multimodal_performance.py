"""
Module 5 Integration Performance Measurement Script.
Measures cold-start vs warm inference latency across all modules and Module 5 fusion.
"""

import time
import os
import sys
import numpy as np
import cv2
import base64
import asyncio

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(root_dir, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.services.face_service import FaceService
from app.services.liveness_service import LivenessService
from app.services.multimodal_fusion_service import MultimodalFusionService, get_deepfake_analyzer
from app.schemas.multimodal import MultimodalAuthRequest


def measure_performance():
    print("=" * 70)
    print("MODULE 5 INTEGRATION PERFORMANCE MEASUREMENT")
    print("=" * 70)

    dummy_bgr = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.ellipse(dummy_bgr, (100, 100), (50, 70), 0, 0, 360, (200, 180, 160), -1)
    dummy_png = cv2.imencode('.png', dummy_bgr)[1].tobytes()
    b64_img = base64.b64encode(dummy_png).decode('utf-8')

    # 1. Module 1 DeepQShield Latency
    analyzer = get_deepfake_analyzer()
    t0 = time.perf_counter()
    res1 = analyzer.analyze_frame_sequence([dummy_bgr])
    t1 = time.perf_counter()
    m1_latency = (t1 - t0) * 1000.0

    # 2. Module 3 Face Liveness Latency
    t0 = time.perf_counter()
    res3 = LivenessService.evaluate_passive_liveness([b64_img])
    t1 = time.perf_counter()
    m3_latency = (t1 - t0) * 1000.0

    # 3. Module 5 End-to-End Orchestration Latency (Warm Runs)
    req = MultimodalAuthRequest(user_id="perf_user_01", face_image_base64=b64_img)

    runs = []
    for _ in range(5):
        t0 = time.perf_counter()
        resp = asyncio.run(MultimodalFusionService.authenticate_multimodal(req))
        t1 = time.perf_counter()
        runs.append((t1 - t0) * 1000.0)

    print(f"Module 1 (DeepQShield) Latency:    {m1_latency:.2f} ms")
    print(f"Module 3 (Face Liveness) Latency:   {m3_latency:.2f} ms")
    print(f"Module 5 End-to-End Fusion Latencies (n=5):")
    print(f"  Min:    {min(runs):.2f} ms")
    print(f"  Avg:    {np.mean(runs):.2f} ms")
    print(f"  Median: {np.median(runs):.2f} ms")
    print(f"  Max:    {max(runs):.2f} ms")
    print("=" * 70)

if __name__ == "__main__":
    measure_performance()
