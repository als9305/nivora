"""
Frame Sampler for Live & Uploaded Video Processing.
Extracts representative frames at configurable sampling rates (fps) or fixed target frame count.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class FrameSampler:
    """
    Configurable Frame Sampler.
    Extracts frames from video files or base64 frame arrays.
    """
    def __init__(self, target_fps: int = 5, max_frames: int = 60):
        self.target_fps = target_fps
        self.max_frames = max_frames

    def sample_from_video_file(self, video_path: str) -> List[np.ndarray]:
        """
        Extract frame arrays from a video file path at target_fps.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sample_step = max(1, int(round(video_fps / self.target_fps)))
        
        frames = []
        frame_count = 0
        
        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_step == 0:
                frames.append(frame)
                
            frame_count += 1
            
        cap.release()
        return frames

    def sample_from_frame_list(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Subsample an existing list of BGR frame arrays down to max_frames limit.
        """
        if len(frames) <= self.max_frames:
            return frames
            
        indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
        return [frames[i] for i in indices]
