"""
Module 5 REST API Router for Multimodal Fusion & Unified Authentication.
Exposes POST /api/v1/authenticate/multimodal for DeepGuard.
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any
from app.schemas.multimodal import MultimodalAuthRequest, MultimodalAuthResponse
from app.services.multimodal_fusion_service import MultimodalFusionService
from app.utils.logging import logger

router = APIRouter(prefix="/api/v1/authenticate", tags=["Multimodal Fusion Orchestrator (Module 5)"])


@router.post("/multimodal", response_model=MultimodalAuthResponse)
async def authenticate_multimodal(req: MultimodalAuthRequest):
    """
    Unified Multimodal Fusion & Authentication Orchestrator Endpoint.
    
    Orchestrates:
    - Module 1 (DeepQShield Deepfake Detection)
    - Module 2 (Face Identity Verification)
    - Module 3 (Face Liveness & Anti-Spoofing)
    - Module 4 (Voice Authentication & Audio PAD)
    - Module 5 (Evidence-Based Multimodal Fusion & Hard Security Gates)
    
    Returns structured evidence, risk classification (LOW/MEDIUM/HIGH), and overall decision.
    """
    try:
        res = await MultimodalFusionService.authenticate_multimodal(req)
        return res
    except Exception as e:
        logger.error(f"Error in multimodal authentication API: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "success": False,
                "error_code": "MULTIMODAL_AUTH_FAILED",
                "message": f"Multimodal authentication service error: {str(e)}"
            }
        )
