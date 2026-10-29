"""
Voice Verification & Audio Anti-Spoofing Model Wrapper for Nivora Backend.
Delegates to models/voice_auth.py VoiceAuthEngine.
"""

import os
import sys
from typing import Dict, Any, Tuple, Optional, List
from app.utils.logging import logger

# Ensure root directory is on sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.voice_auth import voice_auth_engine, AudioPreprocessor


class VoiceModel:
    """
    Voice Verification & Audio Anti-Spoofing Model for Nivora.
    """
    def __init__(self):
        self.engine = voice_auth_engine
        logger.info("VoiceModel initialized with VoiceAuthEngine.")

    def extract_audio_features(self, wav_bytes: bytes) -> Tuple[Optional[List[float]], bool, float]:
        """
        Backward-compatible feature extraction method.
        Returns: (embedding_vector, speech_detected, voice_spoof_probability)
        """
        res = self.engine.process_voice_pipeline(wav_bytes)
        if not res["success"]:
            return None, False, 0.85
        return res["probe_embedding"], True, res["voice_spoof_probability"]

    def verify(self, audio_bytes: bytes, registered_embedding: Optional[List[float]], challenge_phrase: str = "") -> Dict[str, Any]:
        """
        Verifies recorded audio sample against registered speaker embedding & challenge phrase structure.
        """
        res = self.engine.process_voice_pipeline(audio_bytes, registered_embedding)
        return {
            "success": res["success"],
            "speaker_verdict": res["speaker_verdict"],
            "voice_similarity": res["voice_similarity"],
            "liveness_verdict": res["liveness_verdict"],
            "liveness_score": res["liveness_score"],
            "voice_spoof_probability": res["voice_spoof_probability"],
            "attack_type": res.get("attack_type"),
            "final_verdict": res.get("final_verdict", "REJECTED"),
            # There is no validated ASR verifier in this runtime.  This value
            # is informational and must not be reported as a successful match.
            "phrase_match": "UNVALIDATED",
            "speech_detected": res["audio_quality"].get("pass", False),
            "audio_quality": res.get("audio_quality", {}),
            "acoustic_metrics": res.get("acoustic_metrics", {}),
            "message": res.get("message", "Voice verification complete.")
        }


voice_model_instance = VoiceModel()
