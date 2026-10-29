import sys
import os
import glob
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("backend"))
from app.services.liveness_service import LivenessService
from scripts.benchmark_face_liveness import extract_video_frames, encode_frames_to_base64

def main():
    videos = sorted(glob.glob('datasets/genuine/*.mp4')) + sorted(glob.glob('datasets/replay/*.mp4')) + sorted(glob.glob('datasets/borderline/*.mp4'))
    for v in videos:
        frames = extract_video_frames(v)
        b64 = encode_frames_to_base64(frames)
        res = LivenessService.evaluate_passive_liveness(b64)
        label = "GENUINE" if "genuine" in v else "SPOOF"
        print(f"{os.path.basename(v):<30} | True: {label:<7} | Score: {res['score']:.4f} | Verdict: {res['liveness']:<6} | Details: {res.get('details', {})}")

if __name__ == "__main__":
    main()
