from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database.mongodb import connect_to_mongo, close_mongo_connection
from app.ml.model_loader import models
from app.routes import health, auth, verification, dashboard
from app.utils.logging import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup tasks
    logger.info("Initializing DeepGuard Backend Service...")
    await connect_to_mongo()
    models.load_all_models()
    yield
    # Shutdown tasks
    await close_mongo_connection()

app = FastAPI(
    title=settings.APP_NAME,
    description="Deepfake-Resilient Adaptive Multi-Modal Authentication Framework API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration - Set explicit origins and disable wildcard credential conflict
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handlers to satisfy Requirement #20 (never crash)
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred during security processing. System recovered safely.",
            "detail": str(exc) if settings.DEBUG else None
        }
    )

from api.routes import deepfake as deepfake_v1
from api.routes import face as face_v1
from api.routes import liveness as liveness_v1
from api.routes import voice as voice_v1
from api.routes import multimodal as multimodal_v1

# Include API Routers
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(verification.router)
app.include_router(dashboard.router)
app.include_router(deepfake_v1.router)
app.include_router(face_v1.router)
app.include_router(liveness_v1.router)
app.include_router(voice_v1.router)
app.include_router(multimodal_v1.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=settings.DEBUG)
