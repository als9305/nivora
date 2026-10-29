import uuid
import random
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, status
from app.schemas.verification import (
    FaceVerifyRequest, FaceVerifyResponse,
    LivenessChallengeRequest, LivenessChallengeResponse,
    DeepfakeVerifyRequest, DeepfakeVerifyResponse,
    VoiceVerifyRequest, VoiceVerifyResponse,
    AuthenticateRequest, StepUpOTPRequest, StepUpOTPResponse
)
from app.schemas.risk import RiskCalculateRequest, RiskCalculateResponse
from app.services.face_service import FaceService
from app.services.liveness_service import LivenessService
from app.services.deepfake_service import DeepfakeService
from app.services.voice_service import VoiceService
from app.services.device_service import DeviceService
from app.services.risk_engine import RiskEngine
from app.services.explanation_service import ExplanationService
from app.database.repositories import AuthLogRepository
from app.utils.security import create_jwt_token
from app.utils.logging import logger

router = APIRouter(prefix="/api/auth", tags=["Biometric Verification"])

@router.post("/face-verify", response_model=FaceVerifyResponse)
async def face_verify(req: FaceVerifyRequest):
    try:
        img = req.image_base64 or req.live_face_frame_base64 or ""
        result = await FaceService.verify_face(req.user_id, img, req.frames_base64)
        result["similarity_score"] = result.get("face_score", 0.0)
        result["face_match"] = result.get("matched", False)
        return FaceVerifyResponse(**result)
    except Exception as e:
        logger.error(f"Error in face_verify: {e}", exc_info=True)
        return FaceVerifyResponse(
            success=False,
            face_score=0.0,
            similarity_score=0.0,
            face_match=False,
            face_count=0,
            error_code="VERIFICATION_ERROR",
            message=f"Face verification failed: {str(e)}"
        )

@router.post("/liveness", response_model=LivenessChallengeResponse)
async def liveness_challenge(req: LivenessChallengeRequest):
    try:
        result = LivenessService.evaluate_challenge(req.challenge, req.frames_base64)
        return LivenessChallengeResponse(**result)
    except Exception as e:
        logger.error(f"Error in liveness_challenge: {e}", exc_info=True)
        return LivenessChallengeResponse(
            success=False,
            challenge=req.challenge,
            performed=False,
            liveness_score=0.0,
            status="FAIL",
            message=f"Liveness challenge failed: {str(e)}"
        )

@router.post("/deepfake", response_model=DeepfakeVerifyResponse)
async def deepfake_verify(req: DeepfakeVerifyRequest):
    try:
        result = await DeepfakeService.analyze_deepfake(
            req.user_id,
            req.frames_base64,
            req.use_simulation,
            req.simulation_attack_type
        )
        return DeepfakeVerifyResponse(**result)
    except Exception as e:
        logger.error(f"Error in deepfake_verify endpoint: {e}", exc_info=True)
        return DeepfakeVerifyResponse(
            success=False,
            deepfake_probability=1.0,
            label="ERROR",
            confidence=0.0,
            model_mode="error",
            message=f"Deepfake verification failed: {str(e)}"
        )

@router.post("/voice", response_model=VoiceVerifyResponse)
async def voice_verify(req: VoiceVerifyRequest):
    try:
        result = await VoiceService.verify_voice(req.user_id, req.audio_base64, session_id=req.session_id)
        spoof_prob = 1.0 - result.get("liveness_score", 1.0) if result.get("liveness_score") is not None else 0.0
        if "voice_spoof_probability" not in result:
            result["voice_spoof_probability"] = spoof_prob
        pm = result.get("phrase_match")
        result["phrase_match"] = bool(pm is True or pm == "MATCHED")
        if "voice_similarity" not in result:
            result["voice_similarity"] = result.get("similarity_score", 0.0)
        return VoiceVerifyResponse(
            success=result.get("success", False),
            voice_similarity=float(result.get("voice_similarity", 0.0)),
            voice_spoof_probability=float(result.get("voice_spoof_probability", 0.0)),
            phrase_match=result["phrase_match"],
            message=result.get("message", "Voice verification completed.")
        )
    except Exception as e:
        logger.error(f"Error in voice_verify: {e}", exc_info=True)
        return VoiceVerifyResponse(
            success=False,
            voice_similarity=0.0,
            voice_spoof_probability=1.0,
            phrase_match=False,
            message=f"Voice verification failed: {str(e)}"
        )

@router.post("/calculate-risk", response_model=RiskCalculateResponse)
async def calculate_risk(req: RiskCalculateRequest):
    try:
        result = RiskEngine.calculate_risk(
            face_score=req.face_score,
            liveness_score=req.liveness_score,
            deepfake_probability=req.deepfake_probability,
            voice_similarity=req.voice_similarity,
            voice_spoof_probability=req.voice_spoof_probability,
            device_risk=req.device_risk,
            session_risk=req.session_risk
        )
        return RiskCalculateResponse(**result)
    except Exception as e:
        logger.error(f"Error in calculate_risk: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={
            "error_code": "RISK_ENGINE_FAILED",
            "message": "Risk assessment could not be computed."
        }) from e

@router.post("/authenticate")
async def final_authenticate(req: AuthenticateRequest):
    """
    Final composite multi-modal authentication handler.
    Combines all biometric & risk signals, logs attempt, issues token if ALLOWed.
    """
    try:
        dev_risk = DeviceService.calculate_device_risk(req.device_info)
        
        risk_result = RiskEngine.calculate_risk(
            face_score=req.face_score,
            liveness_score=req.liveness_score,
            deepfake_probability=req.deepfake_probability,
            voice_similarity=req.voice_similarity,
            voice_spoof_probability=req.voice_spoof_probability,
            device_risk=dev_risk,
            session_risk=10.0
        )

        verif_data = {
            "face_score": req.face_score,
            "liveness_score": req.liveness_score,
            "deepfake_probability": req.deepfake_probability,
            "voice_similarity": req.voice_similarity,
            "voice_spoof_probability": req.voice_spoof_probability
        }

        explanation = await ExplanationService.generate_explanation(risk_result, verif_data)

        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        token = None

        if risk_result["decision"] == "ALLOW":
            token = create_jwt_token({"sub": req.user_id, "email": req.email, "session_id": session_id})

        log_entry = {
            "session_id": session_id,
            "user_id": req.user_id,
            "email": req.email,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "face_score": req.face_score,
            "liveness_score": req.liveness_score,
            "deepfake_probability": req.deepfake_probability,
            "voice_similarity": req.voice_similarity,
            "voice_spoof_probability": req.voice_spoof_probability,
            "risk_score": risk_result["risk_score"],
            "risk_level": risk_result["risk_level"],
            "decision": risk_result["decision"],
            "reasons": risk_result["reasons"],
            "explanation": explanation,
            "attack_type": req.simulation_type or ("DEEPFAKE_ATTACK" if req.deepfake_probability > 0.6 else "GENUINE")
        }

        log_id = await AuthLogRepository.log_attempt(log_entry)

        return {
            "success": True,
            "session_id": session_id,
            "log_id": log_id,
            "token": token,
            "decision": risk_result["decision"],
            "risk_score": risk_result["risk_score"],
            "risk_level": risk_result["risk_level"],
            "reasons": risk_result["reasons"],
            "score_breakdown": risk_result["score_breakdown"],
            "explanation": explanation,
            "message": f"Authentication finished with decision: {risk_result['decision']}"
        }
    except Exception as e:
        logger.error(f"Error in final_authenticate: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={
            "error_code": "FUSION_FAILED",
            "message": "Authentication decision could not be computed."
        }) from e

@router.post("/step-up", response_model=StepUpOTPResponse)
async def step_up_otp(req: StepUpOTPRequest):
    """
    Step-Up OTP verification for MEDIUM risk cases.
    Accepts 6-digit OTP code (Simulated OTP accepted for hackathon demo).
    """
    if len(req.otp_code) == 6 and req.otp_code.isdigit():
        token = create_jwt_token({"sub": req.user_id, "session_id": req.session_id, "step_up_verified": True})
        return StepUpOTPResponse(
            success=True,
            verified=True,
            decision="ALLOW",
            token=token,
            message="Step-Up OTP verification successful. Access granted."
        )
    else:
        return StepUpOTPResponse(
            success=False,
            verified=False,
            decision="BLOCK",
            token=None,
            message="Invalid OTP code. Step-up authentication failed."
        )

@router.post("/simulate")
async def attack_simulation(simulation_type: str = "DEEPFAKE_ATTACK", email: str = "demo@deepguard.sec"):
    """
    Security Attack Simulation Mode endpoint for immediate hackathon demonstration.
    Presets: GENUINE_USER, PHOTO_ATTACK, REPLAY_ATTACK, DEEPFAKE_ATTACK, VOICE_SPOOF, BORDERLINE_CASE.
    """
    sim_type = simulation_type.upper()

    if sim_type == "GENUINE_USER":
        face, live, fake, v_sim, v_spoof = 0.96, 0.92, 0.04, 0.94, 0.05
    elif sim_type == "PHOTO_ATTACK":
        face, live, fake, v_sim, v_spoof = 0.92, 0.15, 0.35, 0.88, 0.10
    elif sim_type == "REPLAY_ATTACK":
        face, live, fake, v_sim, v_spoof = 0.94, 0.28, 0.65, 0.85, 0.20
    elif sim_type == "DEEPFAKE_ATTACK":
        face, live, fake, v_sim, v_spoof = 0.95, 0.35, 0.93, 0.89, 0.82
    elif sim_type == "VOICE_SPOOF":
        face, live, fake, v_sim, v_spoof = 0.91, 0.88, 0.12, 0.42, 0.91
    elif sim_type == "BORDERLINE_CASE":
        face, live, fake, v_sim, v_spoof = 0.85, 0.72, 0.48, 0.78, 0.35
    else:
        face, live, fake, v_sim, v_spoof = 0.90, 0.90, 0.05, 0.90, 0.05

    auth_req = AuthenticateRequest(
        user_id=f"sim_user_{uuid.uuid4().hex[:6]}",
        email=email,
        face_score=face,
        liveness_score=live,
        deepfake_probability=fake,
        voice_similarity=v_sim,
        voice_spoof_probability=v_spoof,
        simulation_type=sim_type
    )

    res = await final_authenticate(auth_req)
    res["simulation_mode"] = True
    res["attack_type"] = sim_type
    return res
