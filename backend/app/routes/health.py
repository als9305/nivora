from fastapi import APIRouter
from app.database import mongodb
from app.ml.model_loader import models

router = APIRouter(tags=["Health"])

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "DeepGuard Security API",
        "database_connected": mongodb.is_mongodb_connected,
        "models_preloaded": models.is_loaded,
        "version": "1.0.0"
    }
