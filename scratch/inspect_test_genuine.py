import os
import glob
import cv2
import numpy as np
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "backend")

from app.services.liveness_service import LivenessService
from scripts.benchmark_face_liveness import extract_video_frames, encode_frames_to_base64

def inspect_test_genuine():
    genuine_dir = os.path.join("datasets", "genuine")
    videos = sorted(glob.glob(os.path.join(genuine_dir, "*.mp4")))
    print(f"Found {len(videos)} total genuine videos.")
    
    dev_genuine = videos[:3]
    test_genuine = videos[3:]
    
    print("\n--- DEV GENUINE VIDEOS ---")
    for v_path in dev_genuine:
        frames = extract_video_frames(v_path)
        b64 = encode_frames_to_base64(frames)
        res = LivenessService.evaluate_passive_liveness(b64)
        print(f"Video: {os.path.basename(v_path)} | Score: {res['score']:.4f} | Result: {res['liveness']} | Details: {res.get('details', {})}")
        
    print("\n--- HELD-OUT TEST GENUINE VIDEOS ---")
    for v_path in test_genuine:
        frames = extract_video_frames(v_path)
        b64 = encode_frames_to_base64(frames)
        res = LivenessService.evaluate_passive_liveness(b64)
        print(f"Video: {os.path.basename(v_path)} | Score: {res['score']:.4f} | Result: {res['liveness']} | Details: {res.get('details', {})}")

if __name__ == "__main__":
    inspect_test_genuine()
