"""
NIVORA Module 2 Direct Diagnostic Script
Executes standalone direct detector inspection and RGB vs BGR verification.
"""

import os
import sys
import cv2
import numpy as np

# Add project root and backend to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(root_dir, "backend")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from video.face_detector import FaceDetector, global_face_detector
from app.ml.face_model import FaceModel, face_model_instance
from models.face_recognition import FaceRecognitionEngine, face_engine_instance
from app.utils.files import base64_to_cv2, cv2_to_base64


def run_diagnostics():
    detector = global_face_detector
    cfg = detector.get_config()

    print("[NIVORA FACE DETECTOR CONFIG]")
    print(f"Detector: {cfg['detector']}")
    print(f"Input color: {cfg['input_color']}")
    print(f"Min face size: {cfg['min_face_size']}")
    print(f"Thresholds: {cfg['thresholds']}")
    print(f"Device: {cfg['device']}")
    print(f"Detector model loaded: {cfg['detector_loaded']}")
    print(f"Detector device: {cfg['device']}")
    print(f"Model weights available: {cfg['weights_available']}\n")

    # Find candidate image
    test_image_path = None
    possible_paths = [
        "debug_module2_received.jpg",
        "scratch/temp/debug_module2_received.jpg",
        "scratch/temp/debug_face_probe.jpg",
        "data/test/fake/celebdf_fake_05_frame_000.jpg"
    ]
    for p in possible_paths:
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p):
            test_image_path = abs_p
            break

    if not test_image_path:
        print("[ERROR] No test face image found.")
        return

    # Load raw BGR image via OpenCV
    img_bgr = cv2.imread(test_image_path)
    if img_bgr is None:
        print(f"[ERROR] Failed to load image from {test_image_path}")
        return

    h, w = img_bgr.shape[:2]

    # TEST A: BGR Image
    boxes_bgr = detector.detect(img_bgr)
    
    # TEST B: RGB Image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes_rgb = detector.detect(img_rgb)

    print("[NIVORA FACE DETECTOR DIRECT TEST]")
    print(f"Input: {os.path.basename(test_image_path)}")
    print(f"Dimensions: {w} x {h}")
    print(f"Faces detected: {len(boxes_bgr)}")

    print(f"\nFaces detected A (BGR): {len(boxes_bgr)}")
    print(f"Faces detected B (RGB): {len(boxes_rgb)}")

    if boxes_bgr:
        print(f"\n[NIVORA MODULE 2]")
        print(f"Detector faces: {len(boxes_bgr)}")
        bbox = boxes_bgr[0]
        print(f"Face bounding box:")
        print(f"  x={bbox[0]}")
        print(f"  y={bbox[1]}")
        print(f"  width={bbox[2]}")
        print(f"  height={bbox[3]}")

    # Base64 Roundtrip Test
    b64_str = cv2_to_base64(img_bgr)
    b64_len = len(b64_str)
    has_prefix = b64_str.startswith("data:")

    print(f"\n[NIVORA MODULE 2 BACKEND]")
    print(f"Received base64 length: {b64_len}")
    print(f"Data URL prefix detected: {has_prefix}")

    decoded_bgr = base64_to_cv2(b64_str)
    dh, dw, dc = decoded_bgr.shape
    gray = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2GRAY)
    mean_bright = float(np.mean(gray))
    pixel_var = float(np.var(gray))

    print(f"\nAfter decoding:")
    print(f"Decoded dimensions: {dw} x {dh}")
    print(f"Decoded channels: {dc}")
    print(f"Decoded dtype: {decoded_bgr.dtype}")
    print(f"Mean brightness: {mean_bright:.2f}")
    print(f"Pixel variance: {pixel_var:.2f}")

    # Full Pipeline Test
    model = face_model_instance
    pipeline_res = model.process_face_pipeline(decoded_bgr)
    print(f"\n[NIVORA FACE SERVICE]")
    print(f"Quality validation: {'PASS' if pipeline_res['pass'] else 'FAIL'}")
    if pipeline_res.get('quality'):
        print(f"Quality report: {pipeline_res['quality']}")
    if pipeline_res['pass']:
        print(f"Generating InceptionResnetV1 embedding: SUCCESS (dim={len(pipeline_res['embedding'])})")
        print(f"Identity comparison started: READY")

if __name__ == "__main__":
    run_diagnostics()
