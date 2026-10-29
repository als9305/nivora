import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from app.utils.logging import logger

class BaseDeepfakeDetector:
    def predict(self, frames_bgr: List[np.ndarray]) -> Dict[str, Any]:
        raise NotImplementedError

class LightweightMLAdapter(BaseDeepfakeDetector):
    def __init__(self):
        self.face_cascade = None
        try:
            if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data'):
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                cascade = cv2.CascadeClassifier(cascade_path)
                if not cascade.empty():
                    self.face_cascade = cascade
                    logger.info("LightweightMLAdapter face cascade initialized successfully.")
        except Exception as e:
            logger.warning(f"LightweightMLAdapter cascade unavailable: {e}")

    def _analyze_fft_artifacts(self, face_gray: np.ndarray) -> float:
        """FFT analysis for high-frequency spectral artifacts."""
        f = np.fft.fft2(face_gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        
        # High frequency energy ratio
        h, w = face_gray.shape
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 4
        
        y, x = np.ogrid[:h, :w]
        mask = (x - cx)**2 + (y - cy)**2 > radius**2
        
        high_freq_power = np.mean(magnitude_spectrum[mask])
        total_power = np.mean(magnitude_spectrum)
        
        ratio = high_freq_power / (total_power + 1e-8)
        # Synthetic faces often lack high frequency micro-texture or have periodic lattice artifacts
        return float(np.clip((ratio - 0.75) * 2.0, 0.0, 1.0))

    def _analyze_color_gamut_inconsistency(self, face_bgr: np.ndarray) -> float:
        """Detects color boundary anomalies in YCrCb color space."""
        ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        
        std_cr = np.std(cr)
        std_cb = np.std(cb)
        
        # Deepfake swaps often have unnaturally low or high chrominance variance
        diff = abs(std_cr - std_cb)
        return float(np.clip(diff / 25.0, 0.0, 1.0))

    def predict(self, frames_bgr: List[np.ndarray]) -> Dict[str, Any]:
        if not frames_bgr:
            return {
                "deepfake_probability": 0.05,
                "label": "REAL",
                "confidence": 0.85,
                "model_mode": "lightweight_ml",
                "details": "No frames provided"
            }

        frame_scores = []
        for idx, frame in enumerate(frames_bgr):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4) if (self.face_cascade is not None and hasattr(self.face_cascade, 'detectMultiScale') and not self.face_cascade.empty()) else []
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_crop = frame[y:y+h, x:x+w]
                face_gray = gray[y:y+h, x:x+w]
            else:
                face_crop = frame
                face_gray = gray

            face_gray_resized = cv2.resize(face_gray, (128, 128))
            face_bgr_resized = cv2.resize(face_crop, (128, 128))

            fft_risk = self._analyze_fft_artifacts(face_gray_resized)
            color_risk = self._analyze_color_gamut_inconsistency(face_bgr_resized)
            
            frame_risk = 0.6 * fft_risk + 0.4 * color_risk
            frame_scores.append(frame_risk)

        # Temporal aggregation
        avg_score = float(np.mean(frame_scores)) if frame_scores else 0.1
        
        # Determine label & confidence
        label = "DEEPFAKE" if avg_score > 0.50 else "REAL"
        confidence = float(np.abs(avg_score - 0.50) * 2.0 * 0.4 + 0.60)

        return {
            "deepfake_probability": round(avg_score, 3),
            "label": label,
            "confidence": round(confidence, 3),
            "model_mode": "lightweight_ml",
            "frame_scores": [round(s, 3) for s in frame_scores]
        }

class DemoModelAdapter(BaseDeepfakeDetector):
    """Preset model adapter for security attack simulations."""
    def __init__(self, preset_type: str = "DEEPFAKE"):
        self.preset_type = preset_type

    def predict(self, frames_bgr: List[np.ndarray]) -> Dict[str, Any]:
        if self.preset_type == "DEEPFAKE":
            prob = 0.93
            label = "DEEPFAKE"
        elif self.preset_type == "PHOTO_ATTACK":
            prob = 0.72
            label = "DEEPFAKE"
        elif self.preset_type == "BORDERLINE":
            prob = 0.48
            label = "SUSPICIOUS"
        else:
            prob = 0.06
            label = "REAL"

        return {
            "deepfake_probability": prob,
            "label": label,
            "confidence": 0.95,
            "model_mode": "demo_mode",
            "frame_scores": [prob] * max(1, len(frames_bgr))
        }

deepfake_detector_instance = LightweightMLAdapter()
