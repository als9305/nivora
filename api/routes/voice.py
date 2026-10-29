"""
Module 4 REST API Router for Voice Authentication & Audio Anti-Spoofing.
Exposes clean v1 session-bound endpoints for Nivora platform at /api/v1/voice.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from app.services.voice_service import VoiceService, VoiceSessionManager
from app.utils.logging import logger

router = APIRouter(prefix="/api/v1/voice", tags=["Voice Authentication Module"])


class StartVoiceSessionRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="Optional user ID associated with session.")
    purpose: Optional[str] = Field("VERIFY", description="Session purpose: VERIFY or ENROLL.")


class VoiceSessionVerifyRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier to verify against.")
    audio_base64: str = Field(..., description="Base64 encoded live microphone audio recording.")


class VoiceEnrollRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier for speaker voice enrollment.")
    audio_samples_base64: Optional[List[str]] = Field(None, description="List of base64 encoded audio samples for batch enrollment (>= 3).")
    audio_b64: Optional[str] = Field(None, description="Single base64 encoded audio sample for incremental session enrollment.")
    sample_index: Optional[int] = Field(None, description="Sample index (1, 2, or 3) for incremental enrollment.")
    total_samples: Optional[int] = Field(3, description="Total number of required enrollment samples (default 3).")
    session_id: Optional[str] = Field(None, description="Optional enrollment session ID.")


class VoiceVerifyRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier to verify against.")
    audio_base64: str = Field(..., description="Base64 encoded audio sample.")
    session_id: Optional[str] = Field(None, description="Optional session ID.")
    challenge_phrase: Optional[str] = Field(None, description="Optional challenge phrase.")


@router.post("/session/start")
async def start_voice_session(req: Optional[StartVoiceSessionRequest] = None):
    """
    Initiates a new short-lived, anti-replay authentication session.
    Generates a server-side 3-word phonetic challenge phrase (e.g., 'Blue seven river').
    """
    try:
        user_id = req.user_id if req else None
        purpose = req.purpose if req else "VERIFY"
        session_doc = VoiceSessionManager.create_session(user_id=user_id, purpose=purpose)
        return {
            "success": True,
            "session_id": session_doc["session_id"],
            "challenge_phrase": session_doc["challenge_phrase"],
            "expires_in": session_doc["expires_in"],
            "created_at": session_doc["created_at"]
        }
    except Exception as e:
        logger.error(f"Error starting voice session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "message": str(e)}
        )


@router.post("/session/{session_id}/verify")
async def verify_voice_session(session_id: str, req: VoiceSessionVerifyRequest):
    """
    Verifies live microphone audio recording against active session challenge and enrolled speaker template.
    One-time consumption: once verified or expired, the session is invalidated.
    """
    try:
        res = await VoiceService.verify_voice(req.user_id, req.audio_base64, session_id=session_id)
        return res
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in session voice verification: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "error_code": "VERIFICATION_FAILED", "message": str(e)}
        )


@router.post("/enroll")
async def enroll_voice_identity(req: VoiceEnrollRequest):
    """
    Enrolls speaker voice template from multiple audio samples (either batch or incremental sample).
    """
    try:
        if req.audio_samples_base64 and len(req.audio_samples_base64) > 0:
            res = await VoiceService.enroll_voice(req.user_id, req.audio_samples_base64)
        elif req.audio_b64:
            res = await VoiceService.enroll_voice_sample(
                user_id=req.user_id,
                audio_b64=req.audio_b64,
                sample_index=req.sample_index or 1,
                total_samples=req.total_samples or 3,
                session_id=req.session_id
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"success": False, "error_code": "NO_AUDIO_PROVIDED", "message": "No audio data provided."}
            )

        if not res.get("success"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=res
            )
        return res
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in voice enrollment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "error_code": "ENROLLMENT_FAILED", "message": str(e)}
        )


@router.post("/verify")
async def verify_voice_identity(req: VoiceVerifyRequest):
    """
    Verifies probe voice sample against enrolled user template.
    """
    try:
        res = await VoiceService.verify_voice(req.user_id, req.audio_base64, session_id=req.session_id)
        return res
    except Exception as e:
        logger.error(f"Error in voice verification: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "error_code": "VERIFICATION_FAILED", "message": str(e)}
        )


@router.get("/template/{user_id}")
async def get_voice_template_status(user_id: str):
    """
    Queries voice template enrollment status and metadata for user.
    """
    try:
        return await VoiceService.get_template_info(user_id)
    except Exception as e:
        logger.error(f"Error checking voice template status for {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "message": str(e)}
        )


@router.delete("/template/{user_id}")
async def delete_voice_template(user_id: str):
    """
    Deletes enrolled voice template for specified user.
    """
    try:
        return await VoiceService.remove_template(user_id)
    except Exception as e:
        logger.error(f"Error deleting voice template for {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "message": str(e)}
        )


@router.get("/diagnostic")
async def get_voice_diagnostic():
    """
    Temporary development-only diagnostic endpoint reporting checkpoint version,
    scaler status, feature-vector dimension, and loaded model state.
    Never exposes raw audio, embeddings, secrets, or filesystem paths.
    """
    import os
    from models.voice_auth import voice_auth_engine
    net = voice_auth_engine.anti_spoof_net
    return {
        "server_pid": os.getpid(),
        "checkpoint_sha256": net.checkpoint_sha256,
        "model_name": net.model_info.get("model_name"),
        "is_pretrained": net.is_pretrained,
        "scaler_present": net.scaler_mean is not None and net.scaler_std is not None,
        "scaler_dim": int(len(net.scaler_mean)) if net.scaler_mean is not None else 0,
        "feature_dim": 64,
        "tau_liveness": voice_auth_engine.tau_liveness,
        "tau_speaker": voice_auth_engine.tau_speaker,
    }

