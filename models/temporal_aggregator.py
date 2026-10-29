"""
Temporal Aggregation & Suspicious Interval Detection Module for DeepQShield.
Combines frame-level deepfake probabilities into a unified video authenticity score
and detects suspicious temporal intervals across video sessions.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class TemporalAggregator:
    """
    Temporal Score Aggregator.
    Implements multi-frame aggregation algorithms and suspicious interval localization.
    """
    def __init__(
        self,
        strategy: str = "suspicious_weighted",
        suspicious_threshold: float = 0.50,
        high_risk_threshold: float = 0.75,
        real_threshold: float = 0.60,
        fake_threshold: float = 0.70,
    ):
        if not (
            0.0 <= real_threshold < fake_threshold <= 1.0
            and 0.0 <= suspicious_threshold <= 1.0
        ):
            raise ValueError(
                "Expected 0 <= real_threshold < fake_threshold <= 1 and a valid suspicious_threshold"
            )
        self.strategy = strategy
        self.suspicious_threshold = suspicious_threshold
        self.high_risk_threshold = high_risk_threshold
        # This model's held-out real scores are <= 0.4841 and held-out fake
        # scores are >= 0.6958. Keep the webcam abstention band through 0.70
        # between those distributions for live-camera inference, where normal
        # webcam compression can raise a bona-fide score above 0.50. A score in
        # this gap must trigger a retry, never a synthetic-media accusation.
        self.real_threshold = real_threshold
        self.fake_threshold = fake_threshold

    def aggregate(self, frame_scores: List[float], frame_timestamps: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Aggregate frame probabilities and extract suspicious timeline intervals.
        
        Args:
            frame_scores: List of frame-level fake probabilities (0.0 to 1.0)
            frame_timestamps: Optional list of timestamps in seconds corresponding to each frame
            
        Returns:
            Dict containing video_fake_probability, suspicious_frame_ratio, max_fake_probability,
            suspicious_intervals, and quality classification.
        """
        if not frame_scores:
            return {
                "video_fake_probability": 0.05,
                "label": "REAL",
                "confidence": 0.80,
                "frames_analyzed": 0,
                "suspicious_frame_ratio": 0.0,
                "max_fake_probability": 0.05,
                "suspicious_intervals": []
            }
            
        scores = np.array(frame_scores, dtype=np.float32)
        n_frames = len(scores)
        
        # Calculate aggregate video fake probability based on chosen strategy
        if self.strategy == "mean":
            agg_score = float(np.mean(scores))
        elif self.strategy == "median":
            agg_score = float(np.median(scores))
        elif self.strategy == "percentile":
            agg_score = float(np.percentile(scores, 75))
        elif self.strategy == "suspicious_weighted":
            # Weight suspicious frames higher to catch short deepfake manipulations
            suspicious_mask = scores >= self.suspicious_threshold
            if np.any(suspicious_mask):
                suspicious_avg = np.mean(scores[suspicious_mask])
                overall_avg = np.mean(scores)
                agg_score = float(0.7 * suspicious_avg + 0.3 * overall_avg)
            else:
                agg_score = float(np.mean(scores))
        else:
            agg_score = float(np.mean(scores))
            
        # Count suspicious frames
        suspicious_count = int(np.sum(scores >= self.suspicious_threshold))
        suspicious_ratio = float(suspicious_count / n_frames)
        max_score = float(np.max(scores))
        
        # Build timestamps if not provided (assume 5 fps = 0.2s interval)
        if frame_timestamps is None or len(frame_timestamps) != n_frames:
            frame_timestamps = [i * 0.2 for i in range(n_frames)]
            
        # Detect contiguous suspicious intervals
        intervals = self._detect_suspicious_intervals(scores, frame_timestamps)
        
        if agg_score >= self.fake_threshold:
            label = "DEEPFAKE"
        elif agg_score <= self.real_threshold:
            label = "REAL"
        else:
            label = "INDETERMINATE"

        # This is the raw probability of the selected binary class, rather
        # than an artificial floor that can make an uncertain score look
        # decisive.  Calibration on a representative validation set can later
        # replace this value with a calibrated confidence estimate.
        confidence = float(max(agg_score, 1.0 - agg_score))
        
        return {
            "video_fake_probability": round(agg_score, 4),
            "label": label,
            "confidence": round(confidence, 4),
            "decision_thresholds": {
                "real_at_or_below": self.real_threshold,
                "deepfake_at_or_above": self.fake_threshold,
            },
            "frames_analyzed": n_frames,
            "suspicious_frame_count": suspicious_count,
            "suspicious_frame_ratio": round(suspicious_ratio, 4),
            "max_fake_probability": round(max_score, 4),
            "suspicious_intervals": intervals
        }

    def _detect_suspicious_intervals(self, scores: np.ndarray, timestamps: List[float]) -> List[Dict[str, Any]]:
        """Identify start/end time segments where deepfake probabilities exceed threshold."""
        intervals = []
        in_interval = False
        start_time = 0.0
        segment_scores = []
        
        for score, t in zip(scores, timestamps):
            if score >= self.suspicious_threshold:
                if not in_interval:
                    in_interval = True
                    start_time = t
                    segment_scores = [score]
                else:
                    segment_scores.append(score)
            else:
                if in_interval:
                    in_interval = False
                    end_time = t
                    avg_seg_score = float(np.mean(segment_scores))
                    severity = "HIGH" if avg_seg_score >= self.high_risk_threshold else "MEDIUM"
                    intervals.append({
                        "start": round(start_time, 2),
                        "end": round(end_time, 2),
                        "severity": severity,
                        "mean_probability": round(avg_seg_score, 4)
                    })
                    segment_scores = []
                    
        if in_interval:
            end_time = timestamps[-1] + 0.2
            avg_seg_score = float(np.mean(segment_scores))
            severity = "HIGH" if avg_seg_score >= self.high_risk_threshold else "MEDIUM"
            intervals.append({
                "start": round(start_time, 2),
                "end": round(end_time, 2),
                "severity": severity,
                "mean_probability": round(avg_seg_score, 4)
            })
            
        return intervals
