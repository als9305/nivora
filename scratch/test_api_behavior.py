"""
Diagnostic audit script testing API behavior for edge cases.
"""

import os
import sys
import numpy as np
import cv2
import base64
from fastapi.testclient import TestClient

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(root_dir, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.app.main import app

client = TestClient(app)

def b64(img):
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode("utf-8")

def run_api_checks():
    print("--- API BEHAVIOR DIAGNOSTIC CHECKS ---")

    # 1. No Face (blank frame)
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    res_no_face = client.post("/api/v1/face/verify", json={"user_id": "u1", "image_base64": b64(blank)})
    print("1. No Face response:", res_no_face.json())

    # 2. Unknown User
    valid_face = np.full((200, 200, 3), 30, dtype=np.uint8)
    cv2.ellipse(valid_face, (100, 100), (60, 80), 0, 0, 360, (200, 180, 160), -1)
    res_unk = client.post("/api/v1/face/verify", json={"user_id": "unknown_user_999", "image_base64": b64(valid_face)})
    print("2. Unknown User response:", res_unk.json())

    # 3. Invalid Request
    res_inv = client.post("/api/v1/face/verify", json={"user_id": "", "image_base64": "invalid_base64"})
    print("3. Invalid Request response:", res_inv.json())

if __name__ == "__main__":
    run_api_checks()
