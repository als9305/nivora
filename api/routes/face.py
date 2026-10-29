"""
Module 2 REST API Router for Face Identity Enrollment and Verification.
Provides clean v1 endpoints for Nivora platform.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from app.services.face_service import FaceService
from app.utils.logging import logger

router = APIRouter(prefix="/api/v1/face", tags=["Face Identity Module"])


class FaceEnrollRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier for identity enrollment.")
    images_base64: List[str] = Field(..., description="List of base64 encoded face images for enrollment.")


class FaceVerifyV1Request(BaseModel):
    user_id: str = Field(..., description="Unique user identifier to verify against.")
    image_base64: str = Field(..., description="Base64 encoded probe face image.")
    frames_base64: Optional[List[str]] = Field(None, description="Optional short live-camera burst for temporal multi-face validation.")
    session_id: Optional[str] = Field(None, description="Optional authentication session ID for intermediate evidence binding.")


@router.post("/enroll")
async def enroll_face_identity(req: FaceEnrollRequest):
    """
    Enrolls user face identity from one or more sample images.
    Extracts 512-d InceptionResnetV1 embeddings, checks face quality,
    computes L2-normalized template centroid vector, and persists template.
    """
    try:
        res = await FaceService.enroll_face(req.user_id, req.images_base64)
        if not res.get("success"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=res
            )
        return res
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in face enrollment: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "error_code": "ENROLLMENT_FAILED", "message": str(e)}
        )


@router.post("/verify")
async def verify_face_identity(req: FaceVerifyV1Request):
    """
    Verifies probe face image against enrolled user face template.
    Returns exact Cosine Similarity score, decision status (PASS/REJECTED), and quality report.
    """
    try:
        res = await FaceService.verify_face(req.user_id, req.image_base64, req.frames_base64)
        if req.session_id:
            try:
                from app.services.voice_service import VoiceSessionManager
                VoiceSessionManager.store_face_evidence(req.session_id, req.user_id, res)
            except Exception as se:
                logger.debug(f"Could not store face evidence into session: {se}")
        return res
    except Exception as e:
        logger.error(f"Error in face verification: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "error_code": "VERIFICATION_FAILED", "message": str(e)}
        )


@router.get("/template/{user_id}")
async def get_face_template_status(user_id: str):
    """
    Queries face template enrollment status and metadata for user.
    """
    try:
        return await FaceService.get_template_info(user_id)
    except Exception as e:
        logger.error(f"Error checking template status for {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "message": str(e)}
        )


@router.delete("/template/{user_id}")
async def delete_face_template(user_id: str):
    """
    Deletes enrolled face template for specified user.
    """
    try:
        return await FaceService.remove_template(user_id)
    except Exception as e:
        logger.error(f"Error deleting face template for {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"success": False, "message": str(e)}
        )
