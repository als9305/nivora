"""
Unit test for Video Ingestion & End-to-End Analyzer Pipeline.
"""

import sys
import numpy as np

sys.path.insert(0, "d:/projects/DeepGuard")

from video.frame_sampler import FrameSampler
from video.face_detector import FaceDetector
from video.face_tracker import FaceTracker
from video.quality_checker import VideoQualityChecker
from video.analyzer import VideoDeepfakeAnalyzer
from security.signing import verify_deepfake_record


def test_video_pipeline():
    print("Testing Video Ingestion & Deepfake Analysis Pipeline...")
    
    # 1. Create dummy frame sequence (5 frames, 224x224 BGR)
    frames = [(np.random.rand(224, 224, 3) * 255).astype('uint8') for _ in range(5)]
    
    # 2. Test Sampler
    sampler = FrameSampler(target_fps=5, max_frames=3)
    sampled = sampler.sample_from_frame_list(frames)
    assert len(sampled) == 3, f"Frame sampling length mismatch: {len(sampled)}"
    print("[PASS] FrameSampler subsampling")
    
    # 3. Test Detector & Crop
    detector = FaceDetector()
    boxes = detector.detect(frames[0])
    assert len(boxes) >= 1, "Face detector failed to return bounding boxes"
    crop = detector.crop(frames[0], boxes[0])
    assert crop.ndim == 3, "Face crop invalid shape"
    print("[PASS] FaceDetector detect & crop")
    
    # 4. Test Quality Checker
    qc = VideoQualityChecker(min_blur_var=1.0, min_brightness=5.0)
    q_res = qc.check_sequence_quality(frames)
    assert q_res["status"] == "PASS", f"Quality checker failed: {q_res}"
    print("[PASS] VideoQualityChecker status: PASS")
    
    # 5. Test End-to-End Analyzer & Signed PQC Record
    analyzer = VideoDeepfakeAnalyzer(model_checkpoint="checkpoints/deepqshield_best.pth")
    analysis_record = analyzer.analyze_frame_sequence(frames)
    
    assert analysis_record["status"] == "SUCCESS", f"Analyzer failed: {analysis_record}"
    assert "payload" in analysis_record and "integrity" in analysis_record
    print(f"[PASS] VideoDeepfakeAnalyzer Fake Prob: {analysis_record['payload']['result']['video_fake_probability']}")
    
    # 6. Verify PQC Signature on Record
    is_valid, msg = verify_deepfake_record(analysis_record)
    assert is_valid, f"PQC verification failed on video record: {msg}"
    print(f"[PASS] PQC Signature Verified on Video Record: {msg}")
    
    print("\nALL VIDEO PIPELINE UNIT TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_video_pipeline()
