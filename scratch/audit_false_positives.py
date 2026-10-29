import os
import sys
import glob
import cv2
import torch
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "backend")

from app.services.liveness_service import LivenessService
from scripts.benchmark_face_liveness import extract_video_frames, encode_frames_to_base64
from models.face_liveness import MultiDomainFeatureExtractor

def main():
    test_genuine = sorted(glob.glob("datasets/genuine/*.mp4"))[3:]
    test_replay = sorted(glob.glob("datasets/replay/*.mp4"))[3:]
    test_borderline = sorted(glob.glob("datasets/borderline/*.mp4"))[2:]
    
    print("=================================================================")
    print("MODULE 3 HELD-OUT TEST SAMPLE DIAGNOSTIC AUDIT")
    print("=================================================================")
    
    all_test = [(v, 1, "GENUINE_LIVE") for v in test_genuine] + \
               [(v, 0, "REPLAY_ATTACK") for v in test_replay] + \
               [(v, 0, "BORDERLINE_ATTACK") for v in test_borderline]
               
    for v_path, label, category in all_test:
        v_name = os.path.basename(v_path)
        frames = extract_video_frames(v_path, max_frames=15)
        if not frames:
            print(f"Skipping {v_name}: No frames extracted.")
            continue
            
        b64_frames = encode_frames_to_base64(frames)
        res = LivenessService.evaluate_passive_liveness(b64_frames)
        
        # Calculate raw physical features per frame
        fft_list = []
        lbp_list = []
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
            fft_list.append(MultiDomainFeatureExtractor.extract_fft_moire_ratio(gray))
            lbp_h = MultiDomainFeatureExtractor.extract_lbp_histogram(gray)
            lbp_p = lbp_h[lbp_h > 0]
            lbp_list.append(-float(np.sum(lbp_p * np.log2(lbp_p))) if len(lbp_p) > 0 else 0.0)
            
        motion_delta = res.get("details", {}).get("motion_delta", 0.0)
        avg_fft = np.mean(fft_list) if fft_list else 0.0
        avg_lbp = np.mean(lbp_list) if lbp_list else 0.0
        
        is_correct = (res["liveness"] == "LIVE" and label == 1) or (res["liveness"] == "SPOOF" and label == 0)
        status_str = "CORRECT" if is_correct else ("FALSE_POSITIVE (ACCEPTED SPOOF!)" if label == 0 else "FALSE_NEGATIVE (REJECTED LIVE)")
        
        print(f"\nVideo: {v_name:<25} | Category: {category:<17} | True: {label}")
        print(f"  Verdict: {res['liveness']:<6} | Score: {res['score']:.4f} | Threshold: {res['threshold']:.4f} | Status: {status_str}")
        print(f"  Features -> Motion Delta: {motion_delta:.4f} | FFT Ratio: {avg_fft:.4f} | LBP Entropy: {avg_lbp:.4f}")
        print(f"  Reason: {res.get('details', {}).get('reason', '')}")

if __name__ == "__main__":
    main()
