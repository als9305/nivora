"""
Diagnostic Script to Trace Intermediate Values across Liveness Pipeline for Module 3 Correction Pass.
Inspects face crops, neural probability logits, LBP entropy, FFT moiré ratio, chroma variance, and motion delta.
"""

import os
import sys
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(root_dir, "backend")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.services.liveness_service import LivenessService
from models.face_liveness import (
    PassiveAntiSpoofNet,
    MultiDomainFeatureExtractor,
    TemporalLivenessAggregator
)
from video.face_detector import global_face_detector
from video.quality_checker import VideoQualityChecker


def extract_video_frames(video_path: str, max_frames: int = 15) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
    step = max(1, total_frames // max_frames)
    frames = []
    curr = 0
    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if curr % step == 0:
            frames.append(frame)
        curr += 1
    cap.release()
    return frames


def run_diagnostic():
    print("=================================================================")
    print("MODULE 3 DIAGNOSTIC TRACE — PER-VIDEO INTERMEDIATE METRICS")
    print("=================================================================")

    genuine_videos = sorted(glob.glob("datasets/genuine/*.mp4"))
    replay_videos = sorted(glob.glob("datasets/replay/*.mp4"))
    borderline_videos = sorted(glob.glob("datasets/borderline/*.mp4"))

    all_videos = (
        [(v, "GENUINE", 1) for v in genuine_videos] +
        [(v, "REPLAY", 0) for v in replay_videos] +
        [(v, "BORDERLINE", 0) for v in borderline_videos]
    )

    model = LivenessService.get_model()
    aggregator = TemporalLivenessAggregator()
    quality_checker = VideoQualityChecker(min_blur_var=50.0, min_brightness=30.0, max_brightness=230.0, min_face_size=60)

    print(f"{'Video':<28} | {'Type':<10} | {'True':<4} | {'MotionDelta':<12} | {'FFT_Ratio':<10} | {'LBP_Ent':<8} | {'MeanProb':<8} | {'Scores'}")
    print("-" * 110)

    mean_probs_genuine = []
    mean_probs_spoof = []
    motion_deltas_genuine = []
    motion_deltas_spoof = []

    for v_path, v_type, true_label in all_videos:
        v_name = os.path.basename(v_path)
        frames = extract_video_frames(v_path)
        if not frames:
            print(f"{v_name:<28} | {v_type:<10} | {true_label:<4} | COULD_NOT_READ_FRAMES")
            continue

        # Compute Motion Delta
        motion_delta = aggregator.compute_frame_motion_delta(frames)
        if true_label == 1:
            motion_deltas_genuine.append(motion_delta)
        else:
            motion_deltas_spoof.append(motion_delta)

        # Process Face Crops
        face_crops = []
        fft_ratios = []
        lbp_entropies = []
        for frame in frames:
            proc = global_face_detector.process_frame(frame)
            if proc.get("status") == "SINGLE_FACE_ACCEPTED" and proc.get("face_crop") is not None:
                crop = proc["face_crop"]
                face_crops.append(crop)
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
                fft_ratios.append(MultiDomainFeatureExtractor.extract_fft_moire_ratio(gray_crop))
                lbp_h = MultiDomainFeatureExtractor.extract_lbp_histogram(gray_crop)
                lbp_p = lbp_h[lbp_h > 0]
                lbp_entropies.append(-float(np.sum(lbp_p * np.log2(lbp_p))) if len(lbp_p) > 0 else 0.0)

        if not face_crops:
            print(f"{v_name:<28} | {v_type:<10} | {true_label:<4} | Motion={motion_delta:<6.3f} | NO_FACE_LOCATED")
            continue

        avg_fft = float(np.mean(fft_ratios))
        avg_lbp = float(np.mean(lbp_entropies))

        # Pass through model
        model_probs = []
        with torch.no_grad():
            for crop in face_crops:
                resized = cv2.resize(crop, (224, 224))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                normalized = (tensor - mean) / std
                batch = normalized.unsqueeze(0)
                logits = model(batch)
                probs = F.softmax(logits, dim=1)
                live_prob = float(probs[0, 1].item())
                model_probs.append(live_prob)

        mean_prob = float(np.mean(model_probs))
        if true_label == 1:
            mean_probs_genuine.append(mean_prob)
        else:
            mean_probs_spoof.append(mean_prob)

        scores_str = f"[{min(model_probs):.3f} - {max(model_probs):.3f}]"
        print(f"{v_name:<28} | {v_type:<10} | {true_label:<4} | {motion_delta:<12.3f} | {avg_fft:<10.4f} | {avg_lbp:<8.3f} | {mean_prob:<8.4f} | {scores_str}")

    print("\n-----------------------------------------------------------------")
    print("DISTRIBUTION SUMMARY:")
    print("-----------------------------------------------------------------")
    print(f"Genuine User Motion Deltas: min={min(motion_deltas_genuine):.3f}, max={max(motion_deltas_genuine):.3f}, mean={np.mean(motion_deltas_genuine):.3f}")
    print(f"Spoof Attack Motion Deltas: min={min(motion_deltas_spoof):.3f}, max={max(motion_deltas_spoof):.3f}, mean={np.mean(motion_deltas_spoof):.3f}")
    print(f"Genuine User Model Probs (P(LIVE)): min={min(mean_probs_genuine):.4f}, max={max(mean_probs_genuine):.4f}, mean={np.mean(mean_probs_genuine):.4f}")
    print(f"Spoof Attack Model Probs (P(LIVE)): min={min(mean_probs_spoof):.4f}, max={max(mean_probs_spoof):.4f}, mean={np.mean(mean_probs_spoof):.4f}")


if __name__ == "__main__":
    run_diagnostic()
