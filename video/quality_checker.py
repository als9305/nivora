"""
Video & Frame Quality Assessment Engine for DeepQShield.
Validates blur, lighting, resolution, and face dimensions before deepfake inference.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple


class VideoQualityChecker:
    """
    Quality Assurance Engine.
    Filters low-quality frames to prevent false positive deepfake classifications.
    Calibrated min_blur_var = 50.0 (Old arbitrary heuristic: 100.0).
    """
    def __init__(
        self,
        min_blur_var: float = 50.0,
        min_brightness: float = 40.0,
        max_brightness: float = 220.0,
        min_face_size: int = 80
    ):
        self.min_blur_var = min_blur_var
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_face_size = min_face_size

    def check_frame_quality(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Assess quality metrics for a single image frame.
        """
        if frame is None or frame.size == 0:
            return {"pass": False, "reason": "EMPTY_FRAME"}
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        
        # 1. Blur evaluation via Laplacian variance
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_var < self.min_blur_var:
            return {
                "pass": False,
                "reason": "BLURRY_FRAME",
                "blur_variance": round(blur_var, 2),
                "threshold": self.min_blur_var
            }
            
        # 2. Lighting evaluation via mean pixel brightness
        brightness = float(np.mean(gray))
        if brightness < self.min_brightness or brightness > self.max_brightness:
            return {
                "pass": False,
                "reason": "EXTREME_LIGHTING",
                "brightness": round(brightness, 2),
                "range": [self.min_brightness, self.max_brightness]
            }
            
        # 3. Minimum resolution check
        h, w = frame.shape[:2]
        if h < self.min_face_size or w < self.min_face_size:
            return {
                "pass": False,
                "reason": "RESOLUTION_TOO_LOW",
                "dimensions": [w, h],
                "min_required": self.min_face_size
            }
            
        return {
            "pass": True,
            "reason": "QUALITY_PASS",
            "blur_variance": round(blur_var, 2),
            "brightness": round(brightness, 2)
        }

    def check_sequence_quality(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate overall sequence quality across frame batch.
        """
        if not frames:
            return {"status": "QUALITY_TOO_LOW", "reason": "NO_FRAMES"}
            
        pass_count = 0
        reasons = []
        
        for frame in frames:
            res = self.check_frame_quality(frame)
            if res["pass"]:
                pass_count += 1
            else:
                reasons.append(res["reason"])
                
        pass_ratio = pass_count / len(frames)
        if pass_ratio < 0.50:
            return {
                "status": "QUALITY_TOO_LOW",
                "pass_ratio": round(pass_ratio, 2),
                "primary_reason": reasons[0] if reasons else "UNKNOWN"
            }
            
        return {
            "status": "PASS",
            "pass_ratio": round(pass_ratio, 2)
        }
