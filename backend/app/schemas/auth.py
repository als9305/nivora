from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class UserRegisterRequest(BaseModel):
    name: str = Field(..., min_length=1)
    full_name: Optional[str] = None
    username: Optional[str] = None
    email: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)
    face_embedding: Optional[List[float]] = None
    voice_embedding: Optional[List[float]] = None
    image_base64: Optional[str] = None
    face_image: Optional[str] = None
    device_info: Optional[Dict[str, Any]] = None

class FaceEnrollRequest(BaseModel):
    user_id: Optional[str] = None
    email: Optional[str] = None
    image_base64: Optional[str] = None
    face_image: Optional[str] = None
    face_embedding: Optional[List[float]] = None

class UserRegisterResponse(BaseModel):
    success: bool
    user_id: str
    email: str
    name: str
    message: str

class UserLoginRequest(BaseModel):
    email: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)

class UserLoginResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    user_id: Optional[str] = None
    email: Optional[str] = None
    name: Optional[str] = None
    requires_biometrics: bool = True
    challenge_phrase: Optional[str] = None
    liveness_challenges: List[str] = []
    message: str
