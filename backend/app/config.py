try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseModel as BaseSettings
    SettingsConfigDict = None
from typing import Optional

class Settings(BaseSettings):
    APP_NAME: str = "Nivora"
    ENV: str = "development"
    DEBUG: bool = True
    PORT: int = 8000

    JWT_SECRET: str = "deepguard_super_secret_hackathon_key_2026_change_in_prod"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 120

    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "deepguard_db"
    USE_IN_MEMORY_DB_FALLBACK: bool = True

    GEMINI_API_KEY: Optional[str] = ""

    # Biometric Template Protection Key (32-byte secret for AES-GCM AEAD)
    BIOMETRIC_ENCRYPTION_KEY: str = "nivora_aead_biometric_template_secret_key_v1_2026_change_in_prod"

    # Thresholds
    FACE_MATCH_THRESHOLD: float = 0.65
    DEEPFAKE_THRESHOLD: float = 0.50
    VOICE_MATCH_THRESHOLD: float = 0.55

    # Risk Boundaries
    RISK_THRESHOLD_LOW: int = 35
    RISK_THRESHOLD_MEDIUM: int = 70

    # Risk Engine Weights
    WEIGHT_FACE: float = 0.20
    WEIGHT_LIVENESS: float = 0.25
    WEIGHT_DEEPFAKE: float = 0.30
    WEIGHT_VOICE: float = 0.15
    WEIGHT_DEVICE: float = 0.10

    if SettingsConfigDict is not None:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore"
        )
    else:
        model_config = {"extra": "ignore"}

settings = Settings()
