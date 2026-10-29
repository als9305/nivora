from app.ml.face_model import face_model_instance
from app.ml.deepfake_model import deepfake_detector_instance
from app.ml.voice_model import voice_model_instance
from app.utils.logging import logger

class ModelRegistry:
    def __init__(self):
        self.face_model = face_model_instance
        self.deepfake_detector = deepfake_detector_instance
        self.voice_model = voice_model_instance
        self.is_loaded = False

    def load_all_models(self):
        if not self.is_loaded:
            logger.info("Initializing & loading all DeepGuard ML models into memory...")
            # Trigger lazy load / warmups
            _ = self.face_model
            _ = self.deepfake_detector
            _ = self.voice_model
            try:
                from video.analyzer import get_shared_deepfake_analyzer
                _ = get_shared_deepfake_analyzer()
            except Exception as e:
                logger.warning(f"Could not preload shared deepfake analyzer: {e}")
            try:
                from app.services.liveness_service import LivenessService
                _ = LivenessService.get_model()
            except Exception as e:
                logger.warning(f"Could not preload liveness model: {e}")
            self.is_loaded = True
            logger.info("All ML models successfully loaded and ready for high-speed inference!")

models = ModelRegistry()
