"""
DeepQShield FastAPI Routes Module.
Exposes REST endpoints for single image, video sequence, and incremental live session analysis.
"""

import os
import sys
import uuid
import time
import tempfile
import base64
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File, Form
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from api.schemas.deepfake import (
    ImageAnalysisRequest,
    VideoAnalysisRequest,
    LiveSessionStartRequest,
    LiveSessionFrameRequest,
    LiveSessionEndRequest,
    AnalysisResultResponse
)
from video.analyzer import VideoDeepfakeAnalyzer, get_shared_deepfake_analyzer
from app.utils.files import base64_to_cv2
from security.signing import verify_deepfake_record

router = APIRouter(prefix="/api/v1/deepfake", tags=["Deepfake Engine (DeepQShield)"])

# Shared global analyzer instance
_analyzer = get_shared_deepfake_analyzer()

# In-memory transient live analysis sessions store (session_id -> session_dict)
_live_sessions: Dict[str, Dict[str, Any]] = {}
_stored_results: Dict[str, Dict[str, Any]] = {}


@router.get("/model/info", response_model=Dict[str, Any])
async def get_model_info():
    """Return active DeepQShield model metadata, architecture, and PQC security algorithms."""
    return {
        "model_name": _analyzer.model.model_name,
        "version": _analyzer.model.version,
        "backbone": "ResNeXt50_32x4d",
        "enhancement": "LWE / Lattice-Based Feature Enhancement Layer",
        "fusion": "Feature Fusion (4096 -> 2048 projection + residual gate)",
        "attention": "Spatial & Channel Attention (16:1 reduction)",
        "loss_function": "Focal Loss (gamma=2.0, alpha=2.0, label_smoothing=0.1)",
        "pqc_security": {
            "key_exchange": "Kyber1024 KEM",
            "digital_signature": "Dilithium5 Signature",
            "hash_digest": "SHA3-256"
        }
    }


@router.post("/upload", response_model=Dict[str, Any])
async def upload_and_analyze_file(file: UploadFile = File(...)):
    """
    Direct multipart form-data upload for image or video media files.
    Extracts frames and runs DeepQShield analysis + PQC signing.
    """
    filename = file.filename or "media_upload"
    content_type = file.content_type or ""
    contents = await file.read()
    
    if not contents:
        raise HTTPException(status_code=400, detail="EMPTY_FILE_SUBMITTED")
        
    is_video = (
        content_type.startswith("video/") or 
        filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))
    )
    
    frames_bgr = []
    
    if is_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1] or ".mp4") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
            
        try:
            cap = cv2.VideoCapture(tmp_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, frame_count // 30) if frame_count > 0 else 5
            
            curr_idx = 0
            while cap.isOpened() and len(frames_bgr) < 30:
                ret, frame = cap.read()
                if not ret:
                    break
                if curr_idx % step == 0:
                    frames_bgr.append(frame)
                curr_idx += 1
            cap.release()
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
    else:
        # Single image
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is not None:
            frames_bgr.append(img)
            
    if not frames_bgr:
        raise HTTPException(status_code=400, detail="COULD_NOT_DECODE_MEDIA_FRAMES")
        
    record = _analyzer.analyze_frame_sequence(frames_bgr)
    
    # Attach filename and file metadata
    analysis_id = record.get("payload", {}).get("analysis_id", str(uuid.uuid4()))
    if "payload" in record and "input_meta" in record["payload"]:
        record["payload"]["input_meta"]["filename"] = filename
        record["payload"]["input_meta"]["file_size_bytes"] = len(contents)
        record["payload"]["input_meta"]["media_kind"] = "video" if is_video else "image"
        
    _stored_results[analysis_id] = record
    return record


@router.post("/image", response_model=Dict[str, Any])
async def analyze_image(payload: ImageAnalysisRequest):
    """Analyze a single Base64 encoded image frame for deepfake artifacts."""
    try:
        frame_bgr = base64_to_cv2(payload.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"INVALID_IMAGE_DATA: {str(e)}")
        
    record = _analyzer.analyze_frame_sequence([frame_bgr])
    analysis_id = record.get("payload", {}).get("analysis_id", str(uuid.uuid4()))
    _stored_results[analysis_id] = record
    return record


@router.post("/video", response_model=Dict[str, Any])
async def analyze_video(payload: VideoAnalysisRequest):
    """Analyze a sequence of Base64 encoded video frames for deepfake artifacts."""
    if not payload.frames_base64:
        raise HTTPException(status_code=400, detail="NO_FRAMES_PROVIDED")
        
    cv2_frames = []
    for b64 in payload.frames_base64:
        try:
            frame = base64_to_cv2(b64)
            cv2_frames.append(frame)
        except Exception:
            continue
            
    if not cv2_frames:
        raise HTTPException(status_code=400, detail="INVALID_VIDEO_FRAMES")
        
    record = _analyzer.analyze_frame_sequence(cv2_frames)
    analysis_id = record.get("payload", {}).get("analysis_id", str(uuid.uuid4()))
    _stored_results[analysis_id] = record
    if payload.session_id and payload.user_id:
        try:
            from app.services.voice_service import VoiceSessionManager
            VoiceSessionManager.store_deepfake_evidence(payload.session_id, payload.user_id, record)
        except Exception as se:
            pass
    return record


@router.post("/live/start", response_model=Dict[str, Any])
async def start_live_session(payload: LiveSessionStartRequest):
    """Initialize a transient live video deepfake inspection session."""
    session_id = str(uuid.uuid4())
    _live_sessions[session_id] = {
        "session_id": session_id,
        "user_id": payload.user_id,
        "start_time": time.time(),
        "max_duration": payload.max_duration_seconds or 120,
        "frames": [],
        "timestamps": []
    }
    return {
        "status": "SESSION_INITIALIZED",
        "session_id": session_id,
        "message": "Live analysis session started. Submit frames via /live/frame."
    }


@router.post("/live/frame", response_model=Dict[str, Any])
async def ingest_live_frame(payload: LiveSessionFrameRequest):
    """Ingest an incremental video frame into an active live session."""
    session = _live_sessions.get(payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND_OR_EXPIRED")
        
    if time.time() - session["start_time"] > session["max_duration"]:
        _live_sessions.pop(payload.session_id, None)
        raise HTTPException(status_code=408, detail="SESSION_TIMED_OUT")
        
    if len(session["frames"]) >= 100:
        raise HTTPException(status_code=400, detail="MAX_FRAME_LIMIT_REACHED")
        
    try:
        frame = base64_to_cv2(payload.frame_base64)
        session["frames"].append(frame)
        ts = payload.frame_timestamp if payload.frame_timestamp is not None else round(time.time() - session["start_time"], 2)
        session["timestamps"].append(ts)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"FRAME_DECODE_ERROR: {str(e)}")
        
    return {
        "status": "FRAME_ACCEPTED",
        "session_id": payload.session_id,
        "frames_ingested": len(session["frames"])
    }


@router.post("/live/end", response_model=Dict[str, Any])
async def end_live_session(payload: LiveSessionEndRequest):
    """Conclude live analysis session and compute final temporal deepfake decision with PQC signature."""
    session = _live_sessions.pop(payload.session_id, None)
    if not session:
        raise HTTPException(status_code=404, detail="SESSION_NOT_FOUND")
        
    frames = session.get("frames", [])
    if not frames:
        raise HTTPException(status_code=400, detail="NO_FRAMES_SUBMITTED_IN_SESSION")
        
    record = _analyzer.analyze_frame_sequence(frames)
    analysis_id = record.get("payload", {}).get("analysis_id", str(uuid.uuid4()))
    _stored_results[analysis_id] = record
    return record


@router.get("/history", response_model=Dict[str, Any])
async def get_deepfake_history():
    """Retrieve history of all completed DeepQShield analysis records."""
    records = list(_stored_results.values())
    return {
        "status": "SUCCESS",
        "total": len(records),
        "history": records
    }


@router.get("/result/{analysis_id}", response_model=Dict[str, Any])
async def get_analysis_result(analysis_id: str):
    """Retrieve signed analysis result record and verify cryptographic signature."""
    record = _stored_results.get(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="ANALYSIS_RESULT_NOT_FOUND")
        
    res_record = dict(record)
    if "payload" in record and isinstance(record["payload"], dict):
        res_record["payload"] = dict(record["payload"])

    if "integrity" in record and isinstance(record["integrity"], dict):
        is_valid, msg = verify_deepfake_record(record)
        res_record["integrity_verification"] = {
            "valid": is_valid,
            "status": "VERIFIED" if is_valid else "FAILED",
            "message": msg
        }
        integrity = record["integrity"]
        res_record["sha3_256_digest"] = integrity.get("sha3_256_digest")
        res_record["dilithium_signature"] = integrity.get("signature_hex")
        res_record["payload"]["digest_sha3_256"] = integrity.get("sha3_256_digest")
        if "kyber_kem" in record:
            res_record["kyber_kem"] = record["kyber_kem"]
        elif "kyber_kem" in integrity:
            res_record["kyber_kem"] = integrity["kyber_kem"]
    else:
        res_record["integrity_verification"] = {
            "valid": False,
            "status": "NOT_AVAILABLE",
            "message": "Cryptographic operations were not performed for this record."
        }
        res_record["sha3_256_digest"] = None
        res_record["dilithium_signature"] = None
        res_record["kyber_kem"] = None

    return res_record

