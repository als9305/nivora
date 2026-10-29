from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class RiskCalculateRequest(BaseModel):
    face_score: float
    liveness_score: float
    deepfake_probability: float
    voice_similarity: float
    voice_spoof_probability: float
    device_risk: float = 10.0
    session_risk: float = 10.0

class RiskCalculateResponse(BaseModel):
    risk_score: float
    risk_level: str  # LOW, MEDIUM, HIGH
    decision: str    # ALLOW, STEP-UP, BLOCK
    reasons: List[str]
    score_breakdown: Dict[str, float]
    weights_used: Dict[str, float]

class SecurityExplanationResponse(BaseModel):
    threat_summary: str
    evidence: List[str]
    why_decision_made: str
    recommendation: str
    is_llm_generated: bool
