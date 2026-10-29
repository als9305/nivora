import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(root_dir, "backend")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import cv2
import numpy as np
from video.face_detector import global_face_detector

img_path = os.path.join(backend_dir, "debug_module2_received.jpg")
img = cv2.imread(img_path)
assert img is not None

res = global_face_detector.process_frame(img, return_landmarks=True)
lm = res['landmarks']
el, er, n = lm['left_eye'][0], lm['right_eye'][0], lm['nose'][0]
init_yaw = (n - el)/(er - el)
print(f"Base frame: eye_left={el:.1f}, eye_right={er:.1f}, nose={n:.1f}, yaw={init_yaw:.4f}")

for dx in [-25, -15, 0, 15, 25]:
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    r = global_face_detector.process_frame(shifted, return_landmarks=True)
    lm_s = r['landmarks']
    if lm_s:
        el_s, er_s, n_s = lm_s['left_eye'][0], lm_s['right_eye'][0], lm_s['nose'][0]
        y_s = (n_s - el_s)/(er_s - el_s)
        disp = n_s - n
        print(f"dx={dx:+3d}px: nose_x={n_s:.1f}px, disp={disp:+.1f}px, yaw={y_s:.4f}")
