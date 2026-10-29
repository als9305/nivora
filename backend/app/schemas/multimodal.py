"""
Module 5 Multimodal Fusion & Unified Authentication Schemas.
Defines evidence structures, request payloads, and response models.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class FaceIdentityEvidence(BaseModel):
    status: str = Field(..., description="VERIFIED, MISMATCH, NOT_ENROLLED, QUALITY_FAIL, INSUFFICIENT_EVIDENCE")
    similarity: float = Field(0.0, description="Cosine similarity score (0.0 to 1.0)")
    threshold: float = Field(0.65, description="Calibrated match threshold")
    confidence: float = Field(0.0, description="Evidence confidence score")


class FaceLivenessEvidence(BaseModel):
    status: str = Field(..., description="LIVE, SPOOF, INSUFFICIENT_EVIDENCE")
    score: float = Field(0.0, description="Passive Liveness score (0.0 to 1.0)")
    threshold: float = Field(0.50, description="Calibrated liveness threshold")
    attack_type: Optional[str] = Field(None, description="PRINTED_PHOTO, DISPLAY_REPLAY, MASK_ATTACK, None")
    confidence: float = Field(0.0, description="Liveness evidence confidence")


class DeepfakeEvidence(BaseModel):
    status: str = Field(..., description="AUTHENTIC, DEEPFAKE, INDETERMINATE, HIGH_RISK")
    fake_probability: float = Field(0.0, description="Deepfake probability score (0.0 to 1.0)")
    confidence: float = Field(0.0, description="DeepQShield evidence confidence")


class SpeakerEvidence(BaseModel):
    status: str = Field(..., description="VERIFIED, MISMATCH, UNVERIFIED, NOT_ENROLLED, INSUFFICIENT_EVIDENCE")
    similarity: float = Field(0.0, description="Speaker embedding similarity score")
    threshold: float = Field(0.65, description="Calibrated speaker match threshold")


class AudioPADEvidence(BaseModel):
    status: str = Field(..., description="BONAFIDE, SPOOF, INSUFFICIENT_EVIDENCE")
    score: float = Field(0.0, description="Audio Liveness score (0.0 to 1.0)")
    threshold: float = Field(0.70, description="Calibrated audio PAD threshold")
    attack_type: Optional[str] = Field(None, description="REPLAY_ATTACK, SYNTHETIC_TTS, VOICE_CLONE, None")


class MultimodalAuthRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier for identity verification")
    challenge_token: Optional[str] = Field(None, description="Server-generated 300s TTL single-use session challenge token")
    face_image_base64: Optional[str] = Field(None, description="Base64 encoded probe face image frame")
    face_frames_base64: Optional[List[str]] = Field(None, description="List of Base64 encoded face frames for liveness/deepfake tracking")
    audio_pcm_base64: Optional[str] = Field(None, description="Base64 encoded 16 kHz mono PCM WAV audio bytes from microphone")


class MultimodalAuthResponse(BaseModel):
    success: bool = Field(True, description="API execution status")
    decision: str = Field(..., description="AUTHENTICATED, REJECTED, INSUFFICIENT_EVIDENCE, RETRY_REQUIRED")
    overall_confidence: float = Field(..., description="Normalized multimodal confidence score (0.0 to 1.0)")
    risk_level: str = Field(..., description="LOW, MEDIUM, HIGH")
    face_identity: FaceIdentityEvidence
    face_liveness: FaceLivenessEvidence
    deepfake_detection: DeepfakeEvidence
    speaker_identity: SpeakerEvidence
    audio_pad: AudioPADEvidence
    failed_checks: List[str] = Field(default_factory=list, description="List of failed check labels")
    attack_indicators: List[str] = Field(default_factory=list, description="List of detected attack vector tags")
    decision_reasons: List[str] = Field(default_factory=list, description="Human-readable reasons supporting the final decision")
    decision_report: Dict[str, Any] = Field(default_factory=dict, description="Auditable authorization report with control-level evidence")
    processing_time_ms: float = Field(..., description="Total end-to-end multimodal processing latency in ms")
    phrase_match_status: str = Field("UNVALIDATED", description="Phrase matching status (UNVALIDATED)")
    message: str = Field(..., description="User-facing summary message")
