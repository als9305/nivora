"""
End-to-End Video Deepfake Analyzer.
Integrates Sampling -> Detection -> Tracking -> Quality Check -> DeepQShield Inference -> Temporal Aggregation -> PQC Signing.
"""

import os
import sys
import torch
import numpy as np
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from video.frame_sampler import FrameSampler
from video.face_detector import FaceDetector
from video.face_tracker import FaceTracker
from video.quality_checker import VideoQualityChecker
from models.deepqshield import DeepQShieldModel
from models.temporal_aggregator import TemporalAggregator
from training.augmentations import preprocess_image_array
from security.signing import create_signed_deepfake_record


class VideoDeepfakeAnalyzer:
    """
    Production Video Deepfake Inspection Service.
    Loads model instance once and executes transient frame processing.
    """
    def __init__(self, model_checkpoint: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.sampler = FrameSampler(target_fps=5, max_frames=60)
        self.detector = FaceDetector(min_face_size=80)
        self.tracker = FaceTracker(detector=self.detector, policy="reject")
        self.quality_checker = VideoQualityChecker()
        self.aggregator = TemporalAggregator(strategy="suspicious_weighted")
        
        # Instantiate DeepQShield Model
        self.model = DeepQShieldModel(pretrained_backbone=False, seed=42)
        if model_checkpoint and os.path.exists(model_checkpoint):
            self.model.load_checkpoint(model_checkpoint, device=device)
        self.model.to(device)
        self.model.eval()

    def analyze_frame_sequence(self, frames_bgr: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze a sequence of BGR video frames.
        
        Returns:
            Structured, PQC-signed security record
        """
        runtime_evidence = {
            "frames_received": len(frames_bgr),
            "decoded_shapes": [list(frame.shape) for frame in frames_bgr if isinstance(frame, np.ndarray)],
            "decoded_dtype": [str(frame.dtype) for frame in frames_bgr if isinstance(frame, np.ndarray)],
            "color_order": "BGR",
            "jpeg_decode_success": bool(frames_bgr),
            "inference_executed": False,
        }

        # 1. Face Tracking & Crop Extraction
        tracking_res = self.tracker.track_sequence(frames_bgr)
        if tracking_res["status"] == "MULTIPLE_FACES_DETECTED":
            return {
                "status": "MULTIPLE_FACES_DETECTED",
                "message": tracking_res["message"],
                "result": {
                    "classification": "REJECT",
                    "fake_probability": 0.0,
                    "confidence": 0.0
                },
                "quality": {"status": "FAIL", "reason": "MULTIPLE_FACES_DETECTED"},
                "runtime_evidence": runtime_evidence
            }

        if tracking_res["status"] in {"NO_FACE_DETECTED", "FACE_TOO_SMALL_OR_OUTSIDE_VALID_AREA"}:
            return {
                "status": tracking_res["status"],
                "error_code": tracking_res["status"],
                "message": tracking_res["message"],
                "result": {
                    "classification": "INDETERMINATE",
                    "fake_probability": 0.0,
                    "confidence": 0.0,
                },
                "quality": {"status": "FAIL", "reason": tracking_res["status"]},
                "runtime_evidence": runtime_evidence,
            }

        face_crops = tracking_res.get("face_crops", [])

        # 2. Quality Check: evaluate on face crops when available so background wall
        # smoothness does not falsely trigger BLURRY_FRAME on a sharp face.
        eval_targets = face_crops if face_crops else frames_bgr
        frame_quality = [self.quality_checker.check_frame_quality(target) for target in eval_targets]
        pass_count = sum(1 for q in frame_quality if q.get("pass", False))
        pass_ratio = (pass_count / len(eval_targets)) if eval_targets else 0.0
        fail_reasons = [q.get("reason", "UNKNOWN") for q in frame_quality if not q.get("pass", False)]
        
        if not eval_targets:
            quality_res = {"status": "QUALITY_TOO_LOW", "reason": "NO_FRAMES"}
        elif pass_ratio < 0.50:
            quality_res = {
                "status": "QUALITY_TOO_LOW",
                "pass_ratio": round(pass_ratio, 2),
                "primary_reason": fail_reasons[0] if fail_reasons else "UNKNOWN"
            }
        else:
            quality_res = {
                "status": "PASS",
                "pass_ratio": round(pass_ratio, 2)
            }
        runtime_evidence["frame_quality"] = frame_quality

        if quality_res["status"] == "QUALITY_TOO_LOW":
            reason = quality_res.get("primary_reason", "UNKNOWN")
            retry_message = {
                "BLURRY_FRAME": "Camera frame is blurry. Hold still and wait for focus before retrying.",
                "EXTREME_LIGHTING": "Camera frame is too bright or too dark. Adjust lighting and retry.",
                "RESOLUTION_TOO_LOW": "Camera frame is too small for reliable visual security analysis. Move closer and retry.",
            }.get(reason, "Camera frame quality is insufficient. Improve lighting and focus, then retry.")
            return {
                "status": "QUALITY_TOO_LOW",
                "error_code": reason,
                "message": retry_message,
                "result": {
                    "classification": "INDETERMINATE",
                    "fake_probability": 0.0,
                    "confidence": 0.0
                },
                "quality": quality_res,
                "runtime_evidence": runtime_evidence
            }

        # Select only the high-quality crops for model inference
        usable_crops = [crop for crop, quality in zip(face_crops, frame_quality) if quality.get("pass", False)]
        if not usable_crops:
            usable_crops = face_crops
            
        # 3. Model Inference over Frames (Batched forward pass for high performance)
        frame_scores = []
        with torch.inference_mode():
            tensors = [preprocess_image_array(crop) for crop in usable_crops]
            if tensors:
                batch_tensor = torch.cat(tensors, dim=0).to(self.device)
                runtime_evidence["preprocessed_shape"] = list(tensors[0].shape)
                runtime_evidence["preprocessed_dtype"] = str(batch_tensor.dtype)
                runtime_evidence["preprocessed_min"] = float(batch_tensor.min().item())
                runtime_evidence["preprocessed_max"] = float(batch_tensor.max().item())
                runtime_evidence["preprocessed_mean"] = float(batch_tensor.mean().item())
                runtime_evidence["model_input_color_order"] = "RGB"
                logits = self.model(batch_tensor)
                probs = torch.softmax(logits, dim=-1)
                frame_scores = probs[:, 1].tolist()
                
        # 4. Temporal Aggregation
        runtime_evidence["inference_executed"] = True
        temporal_res = self.aggregator.aggregate(frame_scores)
        
        # 5. Construct Metadata & Sign Security Record
        input_meta = {
            "type": "video_frames",
            "frames_received": len(frames_bgr),
            "frames_analyzed": len(usable_crops),
            "frames_discarded_for_quality": len(frames_bgr) - len(usable_crops),
            "quality_status": quality_res["status"],
            "runtime_evidence": runtime_evidence,
        }
        model_info = {
            "name": self.model.model_name,
            "version": self.model.version,
            "architecture": "ResNeXt50_32x4d + LWE + Fusion + Attention"
        }
        
        signed_record = create_signed_deepfake_record(temporal_res, input_meta, model_info)
        signed_record["status"] = "SUCCESS"
        return signed_record


_shared_deepfake_analyzer: Optional[VideoDeepfakeAnalyzer] = None

def get_shared_deepfake_analyzer(model_checkpoint: Optional[str] = "checkpoints/deepqshield_best.pth", device: str = "cpu") -> VideoDeepfakeAnalyzer:
    """Singleton provider for VideoDeepfakeAnalyzer to prevent redundant model instantiation."""
    global _shared_deepfake_analyzer
    if _shared_deepfake_analyzer is None:
        ckpt = model_checkpoint if (model_checkpoint and os.path.exists(model_checkpoint)) else None
        _shared_deepfake_analyzer = VideoDeepfakeAnalyzer(model_checkpoint=ckpt, device=device)
    return _shared_deepfake_analyzer

