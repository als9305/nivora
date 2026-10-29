import os
import sys
import cv2
import base64
import numpy as np
import json
import time

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(root_dir, "backend")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

img_path = os.path.join(backend_dir, "debug_module2_received.jpg")
img = cv2.imread(img_path)
assert img is not None, f"Failed to load probe image at {img_path}"
h, w = img.shape[:2]

def generate_natural_turn_left_sequence(num_frames, max_dx, dy_max=2.0, noise_std=1.0):
    """
    Generates a natural head-turn sequence towards the user's left.
    In mirrored preview, moving to user's left is screen-left (sensor-right, positive dx).
    Includes smooth human acceleration/deceleration S-curve, natural vertical micro-dip,
    and subtle camera sensor noise.
    """
    frames_b64 = []
    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        # S-curve ease-in-out profile
        ease = t * t * (3.0 - 2.0 * t)
        
        dx = float(ease * max_dx)
        dy = float(np.sin(t * np.pi) * dy_max)
        
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Realistic sensor micro-noise
        noise = np.random.normal(0, noise_std, warped.shape).astype(np.int16)
        noisy = np.clip(warped.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        _, buf = cv2.imencode(".jpg", noisy, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        b64_str = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
        frames_b64.append(b64_str)
        
    return frames_b64

test_configs = [
    {
        "test_id": 1,
        "name": "Genuine User 1: Standard Turn Left",
        "num_frames": 15,
        "max_dx": 26.0,
        "dy_max": 2.5,
        "duration_ms": 1820
    },
    {
        "test_id": 2,
        "name": "Genuine User 2: Brisk Quick Turn Left",
        "num_frames": 12,
        "max_dx": 28.0,
        "dy_max": 1.8,
        "duration_ms": 1450
    },
    {
        "test_id": 3,
        "name": "Genuine User 3: Smooth Deliberate Turn Left",
        "num_frames": 18,
        "max_dx": 24.0,
        "dy_max": 3.0,
        "duration_ms": 2180
    },
    {
        "test_id": 4,
        "name": "Genuine User 4: Turn Left with Natural Arc",
        "num_frames": 16,
        "max_dx": 30.0,
        "dy_max": 4.5,
        "duration_ms": 1940
    },
    {
        "test_id": 5,
        "name": "Genuine User 5: Gradual Turn Left with Hold",
        "num_frames": 20,
        "max_dx": 32.0,
        "dy_max": 2.0,
        "duration_ms": 2420
    }
]

print("=" * 90)
print("NIVORA MODULE 3 — 5 GENUINE USER TURN_LEFT TESTS (END-TO-END REST API)")
print("=" * 90)

results = []

for cfg in test_configs:
    tid = cfg["test_id"]
    print(f"\n[{tid}/5] Executing: {cfg['name']}")
    print(f"      Frames: {cfg['num_frames']} | Target Duration: {cfg['duration_ms']} ms | Max Dx: {cfg['max_dx']} px")
    
    frames_b64 = generate_natural_turn_left_sequence(
        num_frames=cfg["num_frames"],
        max_dx=cfg["max_dx"],
        dy_max=cfg["dy_max"]
    )
    
    payload = {
        "frames_base64": frames_b64,
        "user_id": f"genuine_turn_left_user_{tid}",
        "challenge": "TURN_LEFT",
        "challenge_evidence": {
            "duration_ms": cfg["duration_ms"],
            "challenge_id": "TURN_LEFT"
        }
    }
    
    t0 = time.perf_counter()
    resp = client.post("/api/v1/liveness/check", json=payload)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    
    assert resp.status_code == 200, f"HTTP Error {resp.status_code}: {resp.text}"
    data = resp.json()
    
    ch = data.get("challenge", {})
    tm = data.get("temporal_motion", {})
    pl = data.get("passive_liveness", {})
    
    # 12 Required Metrics (Section 14)
    # 1. Challenge (TURN_LEFT)
    # 2. Frames
    # 3. Duration (ms)
    # 4. Landmarks (X/Y)
    # 5. Initial yaw
    # 6. Final yaw
    # 7. Yaw delta
    # 8. Detected direction (LEFT)
    # 9. Challenge result (PASSED)
    # 10. Passive score
    # 11. Temporal evidence
    # 12. Final Module 3 result
    
    entry = {
        "test_id": tid,
        "scenario": cfg["name"],
        "challenge": ch.get("type", "TURN_LEFT"),
        "frames": len(frames_b64),
        "duration_ms": cfg["duration_ms"],
        "landmarks": ch.get("landmarks_detected", "N/A"),
        "initial_yaw": ch.get("initial_yaw", 0.0),
        "final_yaw": ch.get("final_yaw", 0.0),
        "yaw_delta": ch.get("yaw_delta", 0.0),
        "detected_direction": ch.get("detected_direction", "NONE"),
        "challenge_result": ch.get("result", "FAILED"),
        "passive_score": round(data.get("score", 0.0), 4),
        "temporal_evidence": tm.get("evidence", "NONE"),
        "final_module3_result": data.get("liveness", "UNKNOWN"),
        "api_elapsed_ms": elapsed_ms
    }
    results.append(entry)
    
    print(f"      Landmarks:          {entry['landmarks']}")
    print(f"      Initial Yaw:        {entry['initial_yaw']:+.4f}")
    print(f"      Final Yaw:          {entry['final_yaw']:+.4f}")
    print(f"      Yaw Delta:          {entry['yaw_delta']:+.4f}")
    print(f"      Detected Direction: {entry['detected_direction']}")
    print(f"      Challenge Result:   {entry['challenge_result']}")
    print(f"      Passive Score:      {entry['passive_score']:.4f}")
    print(f"      Temporal Evidence:  {entry['temporal_evidence']}")
    print(f"      Final Liveness:     {entry['final_module3_result']}")
    print(f"      Latency:            {elapsed_ms} ms")

print("\n" + "=" * 125)
print("SECTION 14 — COMPLETE 12-METRIC TABULATION")
print("=" * 125)
header = (
    f"{'#':<3} | {'Challenge':<10} | {'Frames':<6} | {'Dur(ms)':<7} | {'Landmarks':<9} | "
    f"{'InitYaw':<8} | {'FinalYaw':<8} | {'YawDelta':<8} | {'Dir':<5} | {'Result':<7} | "
    f"{'Score':<7} | {'Temporal':<8} | {'Final Module 3 Result'}"
)
print(header)
print("-" * 125)

for r in results:
    row = (
        f"{r['test_id']:<3} | {r['challenge']:<10} | {r['frames']:<6} | {r['duration_ms']:<7} | {r['landmarks']:<9} | "
        f"{r['initial_yaw']:<+8.4f} | {r['final_yaw']:<+8.4f} | {r['yaw_delta']:<+8.4f} | {r['detected_direction']:<5} | {r['challenge_result']:<7} | "
        f"{r['passive_score']:<7.4f} | {r['temporal_evidence']:<8} | {r['final_module3_result']}"
    )
    print(row)
print("=" * 125)

out_file = os.path.join(root_dir, "scratch", "turn_left_5_tests_results.json")
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nArtifact saved: {out_file}")
