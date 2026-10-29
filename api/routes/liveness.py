"""
Module 3 REST API Router for Face Liveness & Anti-Spoofing.
Exposes clean v1 endpoint /api/v1/liveness/check for Nivora platform.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from app.services.liveness_service import LivenessService
from app.utils.logging import logger

router = APIRouter(prefix="/api/v1/liveness", tags=["Face Liveness & Anti-Spoofing Module"])


class FaceLivenessCheckRequest(BaseModel):
    frames_base64: List[str] = Field(..., description="List of base64 encoded face video frames or single probe image.")
    user_id: Optional[str] = Field(None, description="Optional user ID for context tracking.")
    session_id: Optional[str] = Field(None, description="Optional authentication session ID for intermediate evidence binding.")
    challenge: Optional[Any] = Field(None, description="Optional active challenge: TURN_LEFT, TURN_RIGHT, BLINK_TWICE, or challenge dict")
    challenge_evidence: Optional[Dict[str, Any]] = Field(None, description="Optional client-side telemetry evidence")


@router.post("/check", response_model=Dict[str, Any])
async def check_face_liveness(req: FaceLivenessCheckRequest):
    """
    Evaluates passive RGB face anti-spoofing, micro-motion temporal liveness, and active challenge-response across frame sequence.
    Detects printed photos, mobile/tablet photo replays, and video replay presentation attacks.
    """
    frames_count = len(req.frames_base64) if req.frames_base64 else 0
    ch_label = req.challenge or "NONE"
    logger.info("[NIVORA MODULE3 BACKEND] Request received")
    logger.info(f"[NIVORA MODULE3 BACKEND] Frames received: {frames_count}")
    logger.info(f"[NIVORA MODULE3 BACKEND] Challenge: {ch_label}")
    print("[NIVORA MODULE3 BACKEND] Request received", flush=True)
    print(f"[NIVORA MODULE3 BACKEND] Frames received: {frames_count}", flush=True)
    print(f"[NIVORA MODULE3 BACKEND] Challenge: {ch_label}", flush=True)
    try:
        res = LivenessService.evaluate_passive_liveness(
            req.frames_base64,
            challenge=req.challenge,
            challenge_evidence=req.challenge_evidence
        )
        if req.session_id and req.user_id:
            try:
                from app.services.voice_service import VoiceSessionManager
                VoiceSessionManager.store_liveness_evidence(req.session_id, req.user_id, res)
            except Exception as se:
                logger.debug(f"Could not store liveness evidence into session: {se}")
        return res
    except Exception as e:
        logger.error(f"Error in face liveness evaluation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "liveness": "INSUFFICIENT_EVIDENCE",
                "score": 0.0,
                "error_code": "LIVENESS_CHECK_FAILED",
                "message": str(e)
            }
        )
