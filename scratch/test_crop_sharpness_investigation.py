"""
NIVORA Module 2 — Crop Sharpness Investigation Script
Analyzes Laplacian variance across:
1. Full frame vs raw face crop vs padded face crop
2. Raw crop vs downscaled crop (160x160)
3. JPEG compression impact
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import io

# Add project root and backend to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(root_dir, "backend")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from video.face_detector import FaceDetector, global_face_detector
from app.ml.face_model import FaceModel, face_model_instance


def compute_laplacian_var(img_bgr: np.ndarray) -> float:
    if img_bgr is None or img_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def run_investigation():
    # Find test image
    test_image_path = None
    possible_paths = [
        "debug_module2_received.jpg",
        "scratch/temp/debug_module2_received.jpg",
        "debug_module2_face_crop.jpg",
        "scratch/temp/debug_face_probe.jpg",
        "data/test/fake/celebdf_fake_05_frame_000.jpg"
    ]
    for p in possible_paths:
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p):
            test_image_path = abs_p
            break

    if not test_image_path:
        print("[ERROR] No image available for investigation.")
        return

    img_bgr = cv2.imread(test_image_path)
    h, w = img_bgr.shape[:2]

    detector = global_face_detector
    boxes = detector.detect(img_bgr)
    if not boxes:
        print("[ERROR] Face detector returned 0 boxes on investigation image.")
        return

    bbox = boxes[0]
    x, y, bw, bh = bbox

    # 1. Raw tight crop (0% margin)
    tight_crop = img_bgr[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)]
    
    # 2. Padded face crop (10% margin — default in FaceDetector)
    pad_crop_10 = detector.crop(img_bgr, bbox, margin_pct=0.10)

    # 3. Padded face crop (20% margin)
    pad_crop_20 = detector.crop(img_bgr, bbox, margin_pct=0.20)

    # 4. Resized face crop (160x160 as used by InceptionResnetV1)
    resized_crop_160 = cv2.resize(pad_crop_10, (160, 160), interpolation=cv2.INTER_AREA)

    # Calculate Laplacian Variances
    full_var = compute_laplacian_var(img_bgr)
    tight_var = compute_laplacian_var(tight_crop)
    crop_10_var = compute_laplacian_var(pad_crop_10)
    crop_20_var = compute_laplacian_var(pad_crop_20)
    resized_160_var = compute_laplacian_var(resized_crop_160)

    ch, cw = pad_crop_10.shape[:2]
    crop_gray = cv2.cvtColor(pad_crop_10, cv2.COLOR_BGR2GRAY)
    crop_brightness = float(np.mean(crop_gray))

    print("========================================================")
    print("NIVORA MODULE 2 CROP SHARPNESS INVESTIGATION REPORT")
    print("========================================================")
    print(f"Original frame: {w}x{h}")
    print(f"Face bounding box: x={x}, y={y}, w={bw}, h={bh}")
    print(f"Raw face crop (10% margin): {cw}x{ch}\n")

    print(f"Full-frame Laplacian variance:     {full_var:.2f}")
    print(f"Tight crop (0% margin) Laplacian:  {tight_var:.2f}")
    print(f"Face crop (10% margin) Laplacian: {crop_10_var:.2f}")
    print(f"Face crop (20% margin) Laplacian: {crop_20_var:.2f}")
    print(f"Resized crop (160x160) Laplacian:  {resized_160_var:.2f}\n")

    print(f"Face-crop brightness: {crop_brightness:.2f} (Required: 40-220)")
    print(f"Face-crop resolution: {cw}x{ch} (Required: >= 80x80)\n")

    print(f"Preprocessing applied before Laplacian calculation: NONE")
    print(f"Laplacian is calculated directly on the raw un-resized face crop array.\n")

    is_sharp = crop_10_var >= 100.0
    print(f"QUALITY RESULT: {'PASS' if is_sharp else 'FAIL (SHARPNESS)'}")

if __name__ == "__main__":
    run_investigation()
