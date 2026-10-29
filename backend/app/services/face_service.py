import time
import os
import sys
import base64
import numpy as np
import cv2
from typing import Dict, Any, Optional, List
from app.ml.face_model import face_model_instance
from app.database.repositories import UserRepository, FaceTemplateRepository
from app.utils.files import base64_to_cv2
from app.config import settings
from app.utils.logging import logger

# Add project root to sys.path for security module
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from security.biometric_crypto import decrypt_template, encrypt_template


class FaceService:
    """
    Production Face Identity Service for Nivora (Module 2).
    Provides face identity template enrollment and real-time verification using AEAD-encrypted templates.
    """

    @staticmethod
    async def enroll_face(user_id: str, images_base64: List[str]) -> Dict[str, Any]:
        """
        Enrolls a user face identity template from one or more sample images.
        Extracts 512-d embeddings, computes centroid, AEAD encrypts, and persists.
        """
        start_time = time.time()
        if not user_id or not user_id.strip():
            return {
                "success": False,
                "error_code": "INVALID_USER_ID",
                "message": "A valid user_id must be provided for face enrollment."
            }

        if not images_base64 or len(images_base64) == 0:
            return {
                "success": False,
                "error_code": "NO_IMAGES_PROVIDED",
                "message": "At least one base64 image frame must be provided for enrollment."
            }

        valid_embeddings = []
        sample_reports = []

        for idx, img_b64 in enumerate(images_base64):
            try:
                img_bgr = base64_to_cv2(img_b64)
                res = face_model_instance.process_face_pipeline(img_bgr)
                sample_reports.append({
                    "sample_index": idx,
                    "pass": res["pass"],
                    "error_code": res.get("error_code"),
                    "quality": res.get("quality", {})
                })

                if res["pass"] and res.get("embedding") is not None:
                    valid_embeddings.append(res["embedding"])
            except Exception as e:
                logger.error(f"Error processing enrollment sample {idx}: {e}")
                sample_reports.append({
                    "sample_index": idx,
                    "pass": False,
                    "error_code": "DECODE_ERROR",
                    "message": str(e)
                })

        if len(valid_embeddings) == 0:
            return {
                "success": False,
                "error_code": "ENROLLMENT_QUALITY_FAILED",
                "message": "All submitted face samples failed face detection or quality checks.",
                "samples_processed": len(images_base64),
                "samples_valid": 0,
                "sample_reports": sample_reports
            }

        # Aggregate valid sample embeddings into normalized centroid template vector
        try:
            template_embedding = face_model_instance.aggregate_embeddings(valid_embeddings)
        except Exception as e:
            return {
                "success": False,
                "error_code": "AGGREGATION_ERROR",
                "message": f"Failed to aggregate template embedding: {str(e)}"
            }

        # Persist template in FaceTemplateRepository (AEAD encrypted)
        template_record = {
            "model_name": "InceptionResnetV1_VGGFace2",
            "embedding_dim": len(template_embedding),
            "embedding": template_embedding,
            "sample_count": len(valid_embeddings),
            "total_submitted": len(images_base64)
        }
        template_id = await FaceTemplateRepository.save_template(user_id, template_record)
        persisted_template = await FaceTemplateRepository.get_template_by_user_id(user_id)
        if not persisted_template or persisted_template.get("template_id") != template_id or "encrypted_template" not in persisted_template:
            logger.error("Face template persistence verification failed for user '%s'.", user_id)
            return {
                "success": False,
                "error_code": "TEMPLATE_PERSISTENCE_FAILED",
                "message": "Face template could not be confirmed in secure storage. Please capture fresh samples and try again.",
            }

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "success": True,
            "user_id": user_id,
            "template_id": template_id,
            "samples_processed": len(images_base64),
            "samples_valid": len(valid_embeddings),
            "model_name": "InceptionResnetV1_VGGFace2",
            "processing_time_ms": elapsed_ms,
            "message": f"Face identity successfully enrolled for user '{user_id}' ({len(valid_embeddings)} valid samples).",
            "sample_reports": sample_reports
        }

    @staticmethod
    async def verify_face(
        user_id: str,
        image_base64: str = "",
        frames_base64: Optional[List[str]] = None,
        predecoded_image_bgr: Optional[np.ndarray] = None,
        predecoded_burst_frames: Optional[List[np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Verifies a probe face image against the user's AEAD-encrypted face template.
        """
        start_time = time.time()
        if not user_id or not user_id.strip():
            return {
                "success": False,
                "matched": False,
                "face_score": 0.0,
                "similarity_score": 0.0,
                "face_count": 0,
                "error_code": "INVALID_USER_ID",
                "message": "A valid user_id must be provided for face verification."
            }

        if predecoded_image_bgr is not None and predecoded_image_bgr.size > 0:
            image_bgr = predecoded_image_bgr
            h, w = image_bgr.shape[:2]
            c = image_bgr.shape[2] if image_bgr.ndim == 3 else 1
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if c == 3 else image_bgr
            mean_bright = float(np.mean(gray))
            pixel_var = float(np.var(gray))
        else:
            b64_len = len(image_base64) if image_base64 else 0
            has_prefix = image_base64.startswith("data:") if image_base64 else False
            logger.info(f"[NIVORA MODULE 2 BACKEND] Received base64 length: {b64_len}")
            logger.info(f"[NIVORA MODULE 2 BACKEND] Data URL prefix detected: {has_prefix}")

            if not image_base64 or b64_len < 100:
                return {
                    "success": False,
                    "matched": False,
                    "face_score": 0.0,
                    "similarity_score": None, # Use None to distinguish from actual 0.0 mismatch
                    "face_count": 0,
                    "error_code": "FACE_FRAME_INVALID",
                    "message": "Camera frame could not be decoded. Image payload missing or invalid."
                }

            try:
                import io
                from PIL import Image

                raw_b64 = image_base64.split(",")[1] if "," in image_base64 else image_base64
                raw_bytes = base64.b64decode(raw_b64)
                pil_img = Image.open(io.BytesIO(raw_bytes))
                fmt = pil_img.format or "JPEG"
                
                # Convert to BGR array for OpenCV
                rgb_arr = np.array(pil_img.convert("RGB"))
                image_bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

                h, w = image_bgr.shape[:2]
                c = image_bgr.shape[2] if image_bgr.ndim == 3 else 1
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if c == 3 else image_bgr
                mean_bright = float(np.mean(gray))
                pixel_var = float(np.var(gray))

                logger.info(f"[NIVORA MODULE 2 BACKEND] Decoded bytes: {len(raw_bytes)}")
                logger.info(f"[NIVORA MODULE 2 BACKEND] Decoded format: {fmt}")
                logger.info(f"[NIVORA MODULE 2 BACKEND] Decoded dimensions: {w} x {h}")
                logger.info(f"[NIVORA MODULE 2 BACKEND] Decoded channels: {c}")
                logger.info(f"[NIVORA MODULE 2 BACKEND] Decoded dtype: {image_bgr.dtype}")
                logger.info(f"[NIVORA MODULE 2 BACKEND] Mean brightness: {mean_bright:.2f}")
                logger.info(f"[NIVORA MODULE 2 BACKEND] Pixel variance: {pixel_var:.2f}")

            except Exception as e:
                logger.error(f"[NIVORA MODULE 2 BACKEND] Base64 decoding failed for user '{user_id}': {e}")
                logger.info("[NIVORA MODULE 2 TRACE] FRAME_DECODED: FAIL")
                return {
                    "success": False,
                    "matched": False,
                    "face_score": None,
                    "similarity_score": None,
                    "face_count": 0,
                    "error_code": "FACE_FRAME_DECODE_FAILED",
                    "message": f"Camera frame could not be decoded: {str(e)}"
                }

        logger.info("[NIVORA MODULE 2 TRACE] FRAME_DECODED: YES")
        logger.info(f"[NIVORA MODULE 2 TRACE] FRAME_WIDTH: {w}")
        logger.info(f"[NIVORA MODULE 2 TRACE] FRAME_HEIGHT: {h}")

        # A short burst lets us distinguish a one-frame false detector box from a
        # real second person.  Never persist this state globally: verification
        # sessions and users must remain isolated.
        has_second_face = False
        if predecoded_burst_frames:
            burst_frames = predecoded_burst_frames[:5]
        else:
            burst_frames = []
            cand_b64 = (frames_base64 or ([image_base64] if image_base64 else []))[:5]
            if len(cand_b64) >= 3:
                for encoded_frame in cand_b64:
                    try:
                        burst_frames.append(base64_to_cv2(encoded_frame))
                    except Exception:
                        continue

        if len(burst_frames) >= 3:
            has_second_face, detected_count = face_model_instance.detector.has_persistent_second_face(burst_frames)
            if has_second_face:
                return {
                    "success": False,
                    "matched": False,
                    "face_score": None,
                    "similarity_score": None,
                    "face_count": detected_count,
                    "error_code": "MULTIPLE_REAL_FACES",
                    "message": "Multiple real faces detected across consecutive camera frames. Only one person may be in view.",
                    "quality": {"pass": False, "reason": "MULTIPLE_REAL_FACES"},
                }

        # Process probe image frame.  A transient secondary detection in the
        # burst is ignored here; each accepted embedding still has exactly one
        # credible primary face after detector filtering.
        proc = face_model_instance.process_face_pipeline(image_bgr)
        # Prefer a clean frame from the burst. This preserves the one-primary-face
        # gate while preventing a single noisy frame from deciding the session.
        if frames_base64 and not proc["pass"] and not has_second_face:
            for candidate in burst_frames[1:]:
                candidate_proc = face_model_instance.process_face_pipeline(candidate)
                if candidate_proc["pass"]:
                    image_bgr, proc = candidate, candidate_proc
                    break
        face_count = proc.get("face_count", 0)
        logger.info(f"[NIVORA MODULE 2 TRACE] FACE_COUNT: {face_count}")

        if not proc["pass"]:
            err_code = proc.get("error_code")
            if face_count == 0:
                standard_err = "FACE_TOO_SMALL" if err_code == "FACE_TOO_SMALL_OR_OUTSIDE_VALID_AREA" else "FACE_NOT_DETECTED"
            elif face_count > 1:
                standard_err = "MULTIPLE_FACES"
            elif err_code in {"FRAME_BLURRY", "FRAME_TOO_BRIGHT_OR_DARK", "FACE_TOO_SMALL"}:
                standard_err = err_code
            else:
                standard_err = "QUALITY_INSUFFICIENT"
            
            logger.warning(f"[NIVORA MODULE 2 BACKEND] Pipeline check failed for user '{user_id}': error_code={err_code}, message={proc.get('message')}")
            logger.info(f"[NIVORA MODULE 2 TRACE] QUALITY_RESULT: FAIL ({standard_err})")
            return {
                "success": False,
                "matched": False,
                "face_score": None,
                "similarity_score": None,
                "face_count": face_count,
                "error_code": standard_err,
                "message": proc.get("message", "Probe image failed quality or face detection check."),
                "quality": proc.get("quality", {})
            }

        probe_embedding = proc["embedding"]
        bbox = proc.get("bbox")
        face_crop_size = f"{bbox[2]}x{bbox[3]}" if bbox else "N/A"
        logger.info(f"[NIVORA MODULE 2 TRACE] FACE_CROP_SIZE: {face_crop_size}")
        logger.info("[NIVORA MODULE 2 TRACE] QUALITY_RESULT: PASS")
        logger.info("[NIVORA MODULE 2 TRACE] EMBEDDING_GENERATED: YES")
        logger.info(f"[NIVORA MODULE 2 TRACE] EMBEDDING_DIMENSION: {len(probe_embedding) if probe_embedding else 0}")

        # Fetch enrolled face template (encrypted)
        template_doc = await FaceTemplateRepository.get_template_by_user_id(user_id)
        if not template_doc:
            logger.info(f"[NIVORA MODULE 2 TRACE] TEMPLATE_LOADED: NO (USER_NOT_ENROLLED: {user_id})")
            return {
                "success": False,
                "matched": False,
                "face_score": None,
                "similarity_score": None,
                "face_count": 1,
                "error_code": "USER_NOT_ENROLLED",
                "message": f"User '{user_id}' has not been enrolled. Please perform face enrollment first."
            }

        logger.info("[NIVORA MODULE 2 TRACE] TEMPLATE_LOADED: YES")

        # Decrypt stored AEAD encrypted biometric template securely in memory
        try:
            if "encrypted_template" in template_doc:
                decrypted_embedding = decrypt_template(template_doc)
            elif "embedding" in template_doc:
                # Backwards fallback for unencrypted legacy templates
                decrypted_embedding = template_doc["embedding"]
            else:
                return {
                    "success": False,
                    "matched": False,
                    "face_score": None,
                    "similarity_score": None,
                    "face_count": 1,
                    "error_code": "CORRUPTED_TEMPLATE",
                    "message": "Stored face template document is corrupted or missing encrypted payload."
                }
        except Exception as e:
            logger.error(f"Template decryption error for user '{user_id}': {e}")
            return {
                "success": False,
                "matched": False,
                "face_score": None,
                "similarity_score": None,
                "face_count": 1,
                "error_code": "TEMPLATE_DECRYPTION_FAILED",
                "message": f"AEAD Biometric Template Decryption Failed: {str(e)}"
            }

        template_dim = len(decrypted_embedding) if decrypted_embedding else 0
        logger.info(f"[NIVORA MODULE 2 TRACE] TEMPLATE_DIMENSION: {template_dim}")

        # Compute empirical cosine similarity
        score = face_model_instance.compute_similarity(probe_embedding, decrypted_embedding)
        
        # Discard decrypted embedding from memory immediately after comparison
        del decrypted_embedding

        threshold = getattr(settings, "FACE_MATCH_THRESHOLD", 0.65)
        matched = bool(score >= threshold)
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        logger.info(f"[NIVORA MODULE 2 TRACE] COSINE_SIMILARITY: {score:.4f}")
        logger.info(f"[NIVORA MODULE 2 TRACE] THRESHOLD: {threshold}")
        logger.info(f"[NIVORA MODULE 2 TRACE] FINAL_FACE_RESULT: {'VERIFIED' if matched else 'FACE_MISMATCH'}")

        return {
            "success": True,
            "user_id": user_id,
            "matched": matched,
            "face_score": round(score, 4),
            "similarity_score": round(score, 4),
            "threshold": threshold,
            "face_count": 1,
            "status": "PASS" if matched else "REJECTED",
            "quality": proc.get("quality", {}),
            "processing_time_ms": elapsed_ms,
            "message": "Face identity verification completed successfully."
        }

    @staticmethod
    async def get_template_info(user_id: str) -> Dict[str, Any]:
        """
        Retrieves template status and metadata for a given user (never exposes raw embedding).
        """
        template_doc = await FaceTemplateRepository.get_template_by_user_id(user_id)
        if not template_doc:
            return {
                "user_id": user_id,
                "enrolled": False,
                "template": None
            }

        return {
            "user_id": user_id,
            "enrolled": True,
            "template": {
                "template_id": template_doc.get("template_id"),
                "model_version": template_doc.get("model_version", "InceptionResnetV1_VGGFace2"),
                "template_version": template_doc.get("template_version", "1.0"),
                "encryption_key_reference": template_doc.get("encryption_key_reference", "ENV_BIOMETRIC_KEY_V1"),
                "encrypted": "encrypted_template" in template_doc,
                "sample_count": template_doc.get("sample_count"),
                "created_at": template_doc.get("created_at"),
                "updated_at": template_doc.get("updated_at")
            }
        }

    @staticmethod
    async def remove_template(user_id: str) -> Dict[str, Any]:
        """
        Deletes enrolled template for a given user.
        """
        deleted = await FaceTemplateRepository.delete_template(user_id)
        return {
            "success": deleted,
            "user_id": user_id,
            "message": f"Face template deleted for user '{user_id}'." if deleted else f"No face template found for user '{user_id}'."
        }
