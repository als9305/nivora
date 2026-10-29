import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(root_dir, "backend")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import cv2
import base64
import numpy as np
import json
import time
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

img_path = os.path.join(backend_dir, "debug_module2_received.jpg")
img = cv2.imread(img_path)
assert img is not None, f"Failed to load real probe frame at {img_path}"

h, w, c = img.shape
print(f"[DIRECT CHALLENGE TEST] Loaded probe image: {w}x{h}x{c}")

# Generate a 15-frame sequence simulating head turning to the user's left (which in camera preview is +dx movement)
frames_b64 = []
for i in range(15):
    dx = int(i * 1.5)  # 0 to 21 pixels shift
    dy = int(np.sin(i * 0.3) * 2)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(img, M, (w, h))
    
    _, buf = cv2.imencode(".jpg", shifted, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    b64_str = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
    frames_b64.append(b64_str)

start_t = time.perf_counter()
payload = {
    "frames_base64": frames_b64,
    "user_id": "test_direct_user",
    "challenge": "TURN_LEFT",
    "challenge_evidence": {
        "duration_ms": 1820,
        "challenge_id": "TURN_LEFT"
    }
}

response = client.post("/api/v1/liveness/check", json=payload)
duration_ms = round((time.perf_counter() - start_t) * 1000, 2)

print("\n" + "="*60)
print(f"[DIRECT CHALLENGE TEST] HTTP Status: {response.status_code}")
print(f"[DIRECT CHALLENGE TEST] Execution Time: {duration_ms} ms")
res_data = response.json()
print("[DIRECT CHALLENGE TEST] Response JSON:")
print(json.dumps({k: v for k, v in res_data.items() if k != "details"}, indent=2))
print("="*60 + "\n")
