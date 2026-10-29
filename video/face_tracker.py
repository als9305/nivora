"""
Face Tracker & Multi-Face Policy Engine.
Tracks primary face across video frames and handles MULTIPLE_FACES_DETECTED states.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from video.face_detector import FaceDetector


class FaceTracker:
    """
    Temporal Face Tracker and Multi-Face policy manager.
    """
    def __init__(self, detector: Optional[FaceDetector] = None, policy: str = "reject"):
        self.detector = detector or FaceDetector()
        self.policy = policy # "reject", "select_primary", or "flag"

    def track_sequence(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Process frame sequence and extract face crops.
        
        Returns:
            Dict containing:
                "status": "OK" or "MULTIPLE_FACES_DETECTED" or "NO_FACE_DETECTED"
                "face_crops": List of cropped face arrays
                "multi_face_counts": List of face counts per frame
        """
        face_crops = []
        counts = []
        any_face_detected = False
        rejected_face_reason = None
        consecutive_multi = 0
        persistent_face_count = 0
        has_persistent_second_face = False
        persistence_threshold = getattr(self.detector, 'multi_face_persistence_frames', 3)
        
        if hasattr(self.detector, 'detect_batch'):
            frames_boxes = self.detector.detect_batch(frames)
        else:
            frames_boxes = [self.detector.detect(f) for f in frames]

        for frame, boxes in zip(frames, frames_boxes):
            num_boxes = len(boxes)
            counts.append(num_boxes)
            
            if num_boxes >= 2:
                consecutive_multi += 1
                persistent_face_count = max(persistent_face_count, num_boxes)
                if consecutive_multi >= persistence_threshold:
                    has_persistent_second_face = True
            else:
                consecutive_multi = 0
                
            if num_boxes >= 1:
                any_face_detected = True
                # For a transient secondary box, retain the largest credible
                # primary face. Persistent multiple faces are rejected below.
                sorted_boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
                crop = self.detector.crop(frame, sorted_boxes[0])
                face_crops.append(crop)
            else:
                rejected_face_reason = rejected_face_reason or self.detector.rejected_face_reason(frame)
                # Do not send a background-only frame into the deepfake model.
                continue
                
        if has_persistent_second_face and self.policy == "reject":
            return {
                "status": "MULTIPLE_FACES_DETECTED",
                "message": "Multiple real faces detected across consecutive frames. Security policy requires one person in view.",
                "face_crops": [],
                "multi_face_counts": counts,
                "persistent_face_count": persistent_face_count,
            }
            
        if not any_face_detected:
            too_small_or_outside = rejected_face_reason == "FACE_TOO_SMALL_OR_OUTSIDE_VALID_AREA"
            return {
                "status": "FACE_TOO_SMALL_OR_OUTSIDE_VALID_AREA" if too_small_or_outside else "NO_FACE_DETECTED",
                "message": (
                    "Face is too small or outside the valid camera area. Move closer and center your face before retrying."
                    if too_small_or_outside else "No face detected in the camera evidence. Center your face and retry."
                ),
                "face_crops": [],
                "multi_face_counts": counts
            }
            
        return {
            "status": "OK",
            "message": f"Successfully tracked {len(face_crops)} face crops.",
            "face_crops": face_crops,
            "multi_face_counts": counts
        }
