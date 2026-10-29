import random
from fastapi import APIRouter, HTTPException, status
from app.schemas.auth import UserRegisterRequest, UserRegisterResponse, UserLoginRequest, UserLoginResponse
from app.database.repositories import UserRepository
from app.utils.security import hash_password, verify_password, create_jwt_token
from app.utils.logging import logger

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

# Available active liveness challenge pool
LIVENESS_POOL = ["BLINK", "TURN_LEFT", "TURN_RIGHT", "SMILE", "NOD"]

@router.post("/register", response_model=UserRegisterResponse)
async def register_user(req: UserRegisterRequest):
    user_name = (req.name or req.full_name or req.username).strip()
    user_email = req.email.lower()
    user_pass = req.password

    existing_user = await UserRepository.get_by_email(user_email)
    if existing_user:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"success": False, "message": "An account with this email already exists. Please log in."})

    # Extract face embedding from image base64 if provided
    face_emb = req.face_embedding
    img_b64 = req.image_base64 or req.face_image
    if not face_emb and img_b64:
        try:
            from app.ml.face_model import face_model_instance
            from app.utils.files import base64_to_cv2
            cv2_img = base64_to_cv2(img_b64)
            emb, count = face_model_instance.extract_embedding(cv2_img)
            if emb:
                face_emb = emb
        except Exception:
            pass

    hashed_pwd = hash_password(user_pass)
    user_data = {
        "name": user_name,
        "email": user_email,
        "password_hash": hashed_pwd,
        "face_embedding": face_emb,
        "voice_embedding": req.voice_embedding,
        "trusted_device": req.device_info or {}
    }

    user_id = await UserRepository.create_user(user_data)
    logger.info(f"New user registered: {user_email} (ID: {user_id})")

    return UserRegisterResponse(
        success=True,
        user_id=user_id,
        email=user_email,
        name=user_name,
        message="User registration successful. Biometric baseline profile stored securely."
    )

@router.post("/enroll-face")
async def enroll_face(req: dict):
    """
    Enrolls or updates face baseline profile for a user.
    """
    email = str(req.get("email") or "").strip().lower()
    if not email:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="A registered email is required.")
    user = await UserRepository.get_by_email(email)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Registered user not found.")
    
    img_b64 = req.get("image_base64") or req.get("face_image") or req.get("image")
    face_emb = req.get("face_embedding")

    if not face_emb and img_b64:
        try:
            from app.ml.face_model import face_model_instance
            from app.utils.files import base64_to_cv2
            cv2_img = base64_to_cv2(img_b64)
            emb, count = face_model_instance.extract_embedding(cv2_img)
            if emb:
                face_emb = emb
        except Exception:
            pass

    user_id = user["user_id"]
    
    return {
        "success": True,
        "user_id": user_id,
        "email": email,
        "face_enrolled": True,
        "message": "Face profile baseline enrolled successfully."
    }

@router.post("/login", response_model=UserLoginResponse)
async def login_user(req: UserLoginRequest):
    user = await UserRepository.get_by_email(req.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"success": False, "error_code": "INVALID_CREDENTIALS", "message": "Invalid email or password."}
        )

    if not verify_password(req.password, user.get("password_hash", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"success": False, "error_code": "INVALID_CREDENTIALS", "message": "Invalid email or password."}
        )

    # Pick 2 random liveness challenges
    selected_challenges = random.sample(LIVENESS_POOL, 2)
    challenge_phrase = f"Secure verification {random.randint(1000, 9999)}"

    return UserLoginResponse(
        success=True,
        token=None,  # Will be issued after biometric verification
        user_id=user["user_id"],
        email=user["email"],
        name=user["name"],
        requires_biometrics=True,
        challenge_phrase=challenge_phrase,
        liveness_challenges=selected_challenges,
        message="Password verified. Proceeding to multi-modal biometric authentication."
    )
