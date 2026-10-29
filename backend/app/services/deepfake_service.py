"""
Deepfake Detection Business Logic Service.
Interfaces DeepGuard authentication workflows with the DeepQShield PyTorch model & PQC signing engine.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from video.analyzer import VideoDeepfakeAnalyzer, get_shared_deepfake_analyzer
from app.utils.files import base64_to_cv2
from app.utils.logging import logger

# Singleton instance of DeepQShield Video Deepfake Analyzer
_analyzer_instance = get_shared_deepfake_analyzer()


class DeepfakeService:
    @staticmethod
    async def analyze_deepfake(
        user_id: str,
        frames_base64: List[str],
        use_simulation: bool = False,
        simulation_attack_type: Optional[str] = None,
        predecoded_frames: Optional[List[np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Extract multi-frame video sequence, execute DeepQShield model inference + temporal aggregation,
        and generate a signed PQC evidence record.
        """
        if use_simulation and simulation_attack_type:
            logger.info(f"Deepfake analysis executing in SIMULATION mode: {simulation_attack_type}")
            # Preset attack simulation values for hackathon studio demo
            if simulation_attack_type == "DEEPFAKE":
                prob = 0.93
                label = "DEEPFAKE"
            elif simulation_attack_type == "PHOTO_ATTACK":
                prob = 0.72
                label = "DEEPFAKE"
            elif simulation_attack_type == "BORDERLINE":
                prob = 0.48
                label = "SUSPICIOUS"
            else:
                prob = 0.06
                label = "REAL"
                
            return {
                "success": True,
                "deepfake_probability": prob,
                "label": label,
                "confidence": 0.95,
                "model_mode": "deepqshield_simulation",
                "message": f"DeepQShield simulation mode execution for attack class: {simulation_attack_type}"
            }

        if not frames_base64 and not predecoded_frames:
            return {
                "success": False,
                "error_code": "DEEPQSHIELD_INSUFFICIENT_EVIDENCE",
                "deepfake_probability": 0.0,
                "label": "INDETERMINATE",
                "confidence": 0.0,
                "model_mode": "deepqshield_v1",
                "message": "No live camera frames were provided for DeepQShield analysis."
            }

        if predecoded_frames:
            cv2_frames = predecoded_frames
        else:
            cv2_frames = []
            for b64 in frames_base64:
                try:
                    frame = base64_to_cv2(b64)
                    cv2_frames.append(frame)
                except Exception:
                    continue

        if not cv2_frames:
            return {
                "success": False,
                "error_code": "DEEPQSHIELD_INSUFFICIENT_EVIDENCE",
                "deepfake_probability": 0.0,
                "label": "INDETERMINATE",
                "confidence": 0.0,
                "model_mode": "deepqshield_v1",
                "message": "Live camera frames could not be decoded for DeepQShield analysis."
            }

        # Run DeepQShield Video Analyzer
        signed_record = _analyzer_instance.analyze_frame_sequence(cv2_frames)
        
        try:
            from api.routes.deepfake import _stored_results
            df_analysis_id = signed_record.get("payload", {}).get("analysis_id")
            if df_analysis_id:
                _stored_results[df_analysis_id] = signed_record
        except Exception:
            pass

        # Extract probability and labels for backwards compatibility with RiskEngine
        payload = signed_record.get("payload", {})
        res = payload.get("result", {})
        if not isinstance(res, dict) or "video_fake_probability" not in res or "label" not in res:
            return {
                "success": False,
                "error_code": "DEEPQSHIELD_INFERENCE_FAILED",
                "deepfake_probability": 0.0,
                "label": "INDETERMINATE",
                "confidence": 0.0,
                "model_mode": "deepqshield_v1",
                "message": "DeepQShield returned an incomplete inference record."
            }
        fake_prob = float(res["video_fake_probability"])
        label = str(res["label"])
        confidence = float(res.get("confidence", 0.0))
        
        return {
            "success": True,
            "deepfake_probability": fake_prob,
            "label": label,
            "confidence": confidence,
            "model_mode": "deepqshield_v1",
            "signed_record": signed_record,
            "message": f"DeepQShield multi-frame temporal analysis complete ({len(cv2_frames)} frames)."
        }
