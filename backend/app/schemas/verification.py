from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class FaceVerifyRequest(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    image_base64: Optional[str] = None
    live_face_frame_base64: Optional[str] = None
    frames_base64: Optional[List[str]] = None

class FaceVerifyResponse(BaseModel):
    success: bool
    face_score: float
    similarity_score: Optional[float] = None
    face_match: bool
    face_count: int = 1
    error_code: Optional[str] = None
    message: str

class LivenessChallengeRequest(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    challenge: str  # BLINK, TURN_LEFT, TURN_RIGHT, SMILE, NOD
    frames_base64: List[str]

class LivenessChallengeResponse(BaseModel):
    success: bool
    challenge: str
    performed: bool
    liveness_score: float
    status: str  # PASS / FAIL
    details: Optional[Dict[str, Any]] = None
    message: str

class DeepfakeVerifyRequest(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    frames_base64: List[str]
    use_simulation: Optional[bool] = False
    simulation_attack_type: Optional[str] = None

class DeepfakeVerifyResponse(BaseModel):
    success: bool
    deepfake_probability: float
    label: str  # REAL / DEEPFAKE
    confidence: float
    model_mode: str  # lightweight_ml, real_ml, demo_mode
    frame_analysis: Optional[List[Dict[str, Any]]] = None
    message: str

class VoiceVerifyRequest(BaseModel):
    user_id: str
    session_id: Optional[str] = None
    audio_base64: str
    challenge_phrase: str

class VoiceVerifyResponse(BaseModel):
    success: bool
    voice_similarity: float
    voice_spoof_probability: float
    phrase_match: bool
    message: str

class AuthenticateRequest(BaseModel):
    user_id: str
    email: str
    face_score: float
    liveness_score: float
    deepfake_probability: float
    voice_similarity: float
    voice_spoof_probability: float
    device_info: Optional[Dict[str, Any]] = None
    simulation_type: Optional[str] = None

class StepUpOTPRequest(BaseModel):
    user_id: str
    session_id: str
    otp_code: str

class StepUpOTPResponse(BaseModel):
    success: bool
    verified: bool
    decision: str
    token: Optional[str] = None
    message: str
