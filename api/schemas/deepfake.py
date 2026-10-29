"""
Pydantic Data Schemas for DeepQShield REST API & Live Sessions.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ImageAnalysisRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image frame string")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier for evidence binding")


class VideoAnalysisRequest(BaseModel):
    frames_base64: List[str] = Field(..., description="List of base64 encoded video frame strings")
    fps: Optional[int] = Field(5, description="Sampling rate frames per second")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier for evidence binding")


class LiveSessionStartRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    max_duration_seconds: Optional[int] = Field(120, description="Max session duration")


class LiveSessionFrameRequest(BaseModel):
    session_id: str = Field(..., description="Active analysis session UUID")
    frame_base64: str = Field(..., description="Base64 encoded video frame string")
    frame_timestamp: Optional[float] = Field(None, description="Timestamp offset in seconds")


class LiveSessionEndRequest(BaseModel):
    session_id: str = Field(..., description="Active analysis session UUID")


class AnalysisResultResponse(BaseModel):
    status: str
    message: str
    record: Optional[Dict[str, Any]] = None
