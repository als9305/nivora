"""
Module 2 — Face Recognition ML Subsystem for Nivora Backend.
Integrates FaceDetector (quality & multi-face gate) and FaceRecognitionEngine (512-d InceptionResnetV1 embedding).
"""

import os
import sys
import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict, Any

# Ensure project root is in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.face_recognition import FaceRecognitionEngine, face_engine_instance
from video.face_detector import FaceDetector, global_face_detector
from video.quality_checker import VideoQualityChecker
from app.utils.logging import logger


class FaceModel:
    """
    Unified Face Recognition Model Subsystem.
    Provides face localization, quality evaluation, 512-d L2 embedding extraction,
    and cosine similarity scoring.
    """
    def __init__(self):
        self.detector = global_face_detector
        self.quality_checker = VideoQualityChecker()
        self.engine = face_engine_instance

    def process_face_pipeline(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Executes end-to-end face processing pipeline:
        1. Multi-face detection gate
        2. Quality check gate (blur, lighting, resolution)
        3. 512-d L2-normalized embedding extraction

        Returns dictionary with complete diagnostic and feature output.
        """
        if image_bgr is None or image_bgr.size == 0:
            logger.warning("[NIVORA MODULE 2 DEBUG] Decoded image is NONE or empty.")
            return {
                "pass": False,
                "error_code": "INVALID_IMAGE",
                "message": "Image frame is empty or invalid.",
                "face_count": 0,
                "embedding": None,
                "quality": {"pass": False, "reason": "EMPTY_FRAME"}
            }

        h, w = image_bgr.shape[:2]
        channels = image_bgr.shape[2] if image_bgr.ndim == 3 else 1
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
        mean_brightness = float(np.mean(gray))
        pixel_var = float(np.var(gray))
        full_blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        logger.info(f"[NIVORA MODULE 2 DEBUG] Decoded image: {w}x{h}, Channels: {channels}, Mean Brightness: {mean_brightness:.2f}, Pixel Variance: {pixel_var:.2f}, Full Blur Var: {full_blur_var:.2f}")

        # Step 1: Detect face & check multi-face policy
        det_result = self.detector.process_frame(image_bgr)
        face_count = det_result["face_count"]

        if face_count == 0:
            rejection_reason = self.detector.rejected_face_reason(image_bgr)
            if rejection_reason:
                return {
                    "pass": False,
                    "error_code": rejection_reason,
                    "message": "Face is too small or outside the valid camera area. Move closer and center your face before retrying.",
                    "face_count": 0,
                    "embedding": None,
                    "quality": {"pass": False, "reason": rejection_reason},
                }
            logger.info(
                "[NIVORA MODULE 2 QUALITY DEBUG]\n\n"
                "Faces detected: 0\n\n"
                "Face bounding box:\n"
                "x=None\n"
                "y=None\n"
                "width=None\n"
                "height=None\n\n"
                "Face crop dimensions:\n"
                "width=None\n"
                "height=None\n\n"
                "Face crop Laplacian variance=N/A\n"
                "Required >= 100\n\n"
                "Face crop brightness=N/A\n"
                "Required 40–220\n\n"
                "Face crop resolution=N/A\n"
                "Required >= 80x80\n\n"
                "Quality result=\n"
                "FAIL (NO_FACE)\n"
            )
            return {
                "pass": False,
                "error_code": "NO_FACE",
                "message": "No face detected in input frame.",
                "face_count": 0,
                "embedding": None,
                "quality": {"pass": False, "reason": "NO_FACE"}
            }

        if face_count > 1:
            # Quality takes precedence over a multi-face verdict.  In a blown-out,
            # dark, or blurred frame, a secondary detector box is not reliable
            # evidence of another person and the user needs a camera retry prompt.
            candidate_boxes = self.detector.detect(image_bgr)
            primary_quality = None
            if candidate_boxes:
                primary_bbox = max(candidate_boxes, key=lambda box: box[2] * box[3])
                primary_quality = self.quality_checker.check_frame_quality(
                    self.detector.crop(image_bgr, primary_bbox)
                )
            if primary_quality is not None and not primary_quality["pass"]:
                reason = primary_quality.get("reason", "QUALITY_INSUFFICIENT")
                quality_code = {
                    "BLURRY_FRAME": "FRAME_BLURRY",
                    "EXTREME_LIGHTING": "FRAME_TOO_BRIGHT_OR_DARK",
                    "RESOLUTION_TOO_LOW": "FACE_TOO_SMALL",
                }.get(reason, "QUALITY_INSUFFICIENT")
                return {
                    "pass": False,
                    "error_code": quality_code,
                    "message": self._quality_retry_message(reason),
                    "face_count": face_count,
                    "embedding": None,
                    "quality": primary_quality,
                }
            bbox = det_result.get("bbox")
            bx = bbox[0] if bbox else "None"
            by = bbox[1] if bbox else "None"
            bw = bbox[2] if bbox else "None"
            bh = bbox[3] if bbox else "None"
            logger.info(
                f"[NIVORA MODULE 2 QUALITY DEBUG]\n\n"
                f"Faces detected: {face_count}\n\n"
                f"Face bounding box:\n"
                f"x={bx}\n"
                f"y={by}\n"
                f"width={bw}\n"
                f"height={bh}\n\n"
                f"Quality result=\n"
                f"FAIL (MULTIPLE_FACES)\n"
            )
            return {
                "pass": False,
                "error_code": "MULTIPLE_FACES",
                "message": f"Multiple faces detected ({face_count}). Single user face required.",
                "face_count": face_count,
                "embedding": None,
                "quality": {"pass": False, "reason": "MULTIPLE_FACES"}
            }

        bbox = det_result["bbox"]
        x, y, bw, bh = bbox
        face_crop = det_result["face_crop"]
        ch, cw = face_crop.shape[:2]
        
        # Save exact face crop image for local development inspection
        try:
            cv2.imwrite("debug_module2_face_crop.jpg", face_crop)
            os.makedirs("scratch/temp", exist_ok=True)
            cv2.imwrite("scratch/temp/debug_module2_face_crop.jpg", face_crop)
        except Exception as se:
            logger.warning(f"Could not save debug face crop: {se}")

        crop_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
        crop_blur_var = float(cv2.Laplacian(crop_gray, cv2.CV_64F).var())
        crop_brightness = float(np.mean(crop_gray))

        # Calculate tight crop (unpadded bbox) for comparative sharpness analysis
        tight_crop = image_bgr[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]
        if tight_crop.size > 0:
            tight_gray = cv2.cvtColor(tight_crop, cv2.COLOR_BGR2GRAY) if tight_crop.ndim == 3 else tight_crop
            tight_blur_var = float(cv2.Laplacian(tight_gray, cv2.CV_64F).var())
        else:
            tight_blur_var = crop_blur_var

        # Step 2: Quality Check Gate
        quality_res = self.quality_checker.check_frame_quality(face_crop)
        quality_pass = quality_res["pass"]

        failing_cond = None
        if not quality_pass:
            raw_reason = quality_res.get("reason", "")
            if raw_reason == "BLURRY_FRAME" or raw_reason == "SHARPNESS":
                failing_cond = "SHARPNESS"
            elif raw_reason == "EXTREME_LIGHTING" or raw_reason == "BRIGHTNESS":
                failing_cond = "BRIGHTNESS"
            elif raw_reason == "RESOLUTION_TOO_LOW" or raw_reason == "RESOLUTION":
                failing_cond = "RESOLUTION"
            else:
                failing_cond = raw_reason
            quality_result_str = f"FAIL ({failing_cond})"
        else:
            quality_result_str = "PASS"

        logger.info(
            f"[NIVORA MODULE 2 CROP DIAGNOSTIC]\n\n"
            f"Original frame:\n"
            f"{w}x{h}\n\n"
            f"Face bounding box:\n"
            f"x={x}, y={y}, w={bw}, h={bh}\n\n"
            f"Raw face crop:\n"
            f"{cw}x{ch}\n\n"
            f"Full-frame Laplacian:\n"
            f"{full_blur_var:.2f}\n\n"
            f"Face-crop Laplacian:\n"
            f"{crop_blur_var:.2f}\n\n"
            f"Tight-crop Laplacian:\n"
            f"{tight_blur_var:.2f}\n\n"
            f"Face-crop brightness:\n"
            f"{crop_brightness:.2f}\n\n"
            f"Face-crop resolution:\n"
            f"{cw}x{ch}\n\n"
            f"Any preprocessing:\n"
            f"NO (Laplacian calculated directly on raw un-resized face crop array)\n\n"
            f"QUALITY RESULT:\n"
            f"{quality_result_str}\n"
        )

        logger.info(
            f"[NIVORA MODULE 2 QUALITY DEBUG]\n\n"
            f"Faces detected: 1\n\n"
            f"Face bounding box:\n"
            f"x={x}\n"
            f"y={y}\n"
            f"width={bw}\n"
            f"height={bh}\n\n"
            f"Face crop dimensions:\n"
            f"width={cw}\n"
            f"height={ch}\n\n"
            f"Face crop Laplacian variance={crop_blur_var:.2f}\n"
            f"Required >= 50.0 (Calibrated)\n\n"
            f"Face crop brightness={crop_brightness:.2f}\n"
            f"Required 40–220\n\n"
            f"Face crop resolution={cw}x{ch}\n"
            f"Required >= 80x80\n\n"
            f"Quality result=\n"
            f"{quality_result_str}\n"
        )

        if not quality_pass:
            if failing_cond == "SHARPNESS":
                err_msg = self._quality_retry_message("BLURRY_FRAME")
            elif failing_cond == "BRIGHTNESS":
                err_msg = self._quality_retry_message("EXTREME_LIGHTING")
            elif failing_cond == "RESOLUTION":
                err_msg = "Face resolution is too low. Move closer to the camera."
            else:
                err_msg = f"Face crop failed quality check: {failing_cond}"

            return {
                "pass": False,
                "error_code": {
                    "SHARPNESS": "FRAME_BLURRY",
                    "BRIGHTNESS": "FRAME_TOO_BRIGHT_OR_DARK",
                    "RESOLUTION": "FACE_TOO_SMALL",
                }.get(failing_cond, failing_cond),
                "message": err_msg,
                "face_count": 1,
                "embedding": None,
                "quality": {
                    "pass": False,
                    "reason": failing_cond,
                    "crop_blur_var": crop_blur_var,
                    "crop_brightness": crop_brightness,
                    "crop_dimensions": [cw, ch]
                }
            }

        # Step 3: Extract 512-d L2-normalized embedding
        try:
            embedding_arr = self.engine.extract_embedding(face_crop)
            embedding_list = embedding_arr.tolist()
            return {
                "pass": True,
                "error_code": None,
                "message": "Face embedding extracted successfully.",
                "face_count": 1,
                "bbox": det_result["bbox"],
                "embedding": embedding_list,
                "quality": quality_res
            }
        except Exception as e:
            logger.error(f"Error extracting face embedding: {e}", exc_info=True)
            return {
                "pass": False,
                "error_code": "EMBEDDING_EXTRACTION_ERROR",
                "message": f"Failed to compute face embedding: {str(e)}",
                "face_count": 1,
                "embedding": None,
                "quality": quality_res
            }

    def compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Computes Cosine Similarity between two 512-d face vectors."""
        return self.engine.compute_cosine_similarity(vec1, vec2)

    @staticmethod
    def _quality_retry_message(reason: str) -> str:
        if reason == "BLURRY_FRAME":
            return "Camera frame is blurry. Hold still and wait for the camera to focus before retrying."
        if reason == "EXTREME_LIGHTING":
            return "Camera frame is too bright or too dark. Adjust lighting so your face is evenly illuminated, then retry."
        if reason == "RESOLUTION_TOO_LOW":
            return "Face is too small for secure verification. Move closer to the camera and retry."
        return "Camera frame quality is insufficient. Improve lighting and focus, then retry."

    def aggregate_embeddings(self, embeddings: List[List[float]]) -> List[float]:
        """Aggregates multiple face sample embeddings into normalized template vector."""
        return self.engine.aggregate_template_embeddings(embeddings)


face_model_instance = FaceModel()
