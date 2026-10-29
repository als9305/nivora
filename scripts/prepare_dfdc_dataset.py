"""
DFDC & DeepGuard Dataset Extractor & Splitter.
Extracts 224x224 face crops from MP4 videos in datasets/ with strict video/identity isolation across train, val, and test splits.
"""

import os
import sys
import glob
import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, "d:/projects/DeepGuard")

from video.frame_sampler import FrameSampler
from video.face_detector import FaceDetector


def prepare_dataset():
    data_base = "data"
    # Clear existing synthetic placeholder images in data/
    splits = ["train", "val", "test"]
    labels = ["real", "fake"]
    
    for s in splits:
        for l in labels:
            d = os.path.join(data_base, s, l)
            os.makedirs(d, exist_ok=True)
            for f in glob.glob(os.path.join(d, "*.*")):
                try:
                    os.remove(f)
                except Exception:
                    pass

    sampler = FrameSampler(target_fps=5, max_frames=50)
    detector = FaceDetector(min_face_size=60)
    
    # Define strict video split mappings
    split_mapping = {
        "train": {
            "fake": [
                "dfdc_fake_01.mp4", "dfdc_fake_02.mp4", "dfdc_fake_03.mp4",
                "dfdc_fake_04.mp4", "dfdc_fake_05.mp4", "dfdc_fake_06.mp4",
                "celebdf_fake_01.mp4", "celebdf_fake_02.mp4", "celebdf_fake_03.mp4",
                "ffpp_fake_01.mp4", "ffpp_fake_02.mp4", "ffpp_fake_03.mp4",
                "wilddf_fake_01.mp4", "wilddf_fake_02.mp4", "wilddf_fake_03.mp4"
            ],
            "real": [
                "genuine_user_01.mp4", "genuine_user_02.mp4", "genuine_user_03.mp4"
            ]
        },
        "val": {
            "fake": [
                "dfdc_fake_07.mp4", "dfdc_fake_08.mp4",
                "celebdf_fake_04.mp4", "ffpp_fake_04.mp4", "wilddf_fake_04.mp4"
            ],
            "real": [
                "genuine_user_04.mp4"
            ]
        },
        "test": {
            "fake": [
                "dfdc_fake_09.mp4", "dfdc_fake_10.mp4",
                "celebdf_fake_05.mp4", "ffpp_fake_05.mp4", "wilddf_fake_05.mp4"
            ],
            "real": [
                "genuine_user_05.mp4", "real_webcam_face.mp4"
            ]
        }
    }
    
    counts = {"train": {"real": 0, "fake": 0}, "val": {"real": 0, "fake": 0}, "test": {"real": 0, "fake": 0}}
    
    for split, label_dict in split_mapping.items():
        for label, video_files in label_dict.items():
            folder_category = "genuine" if label == "real" else "deepfake"
            target_dir = os.path.join(data_base, split, label)
            os.makedirs(target_dir, exist_ok=True)
            
            for vfile in video_files:
                vpath = os.path.join("datasets", folder_category, vfile)
                if not os.path.exists(vpath):
                    print(f"[WARN] Video file missing: {vpath}")
                    continue
                    
                frames = sampler.sample_from_video_file(vpath)
                v_basename = os.path.splitext(vfile)[0]
                
                for idx, frame in enumerate(frames):
                    bboxes = detector.detect(frame)
                    if len(bboxes) > 0:
                        crop_bgr = detector.crop(frame, bboxes[0])
                    else:
                        crop_bgr = frame
                        
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    crop_resized = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_AREA)
                    
                    img_name = f"{v_basename}_frame_{idx:03d}.jpg"
                    img_path = os.path.join(target_dir, img_name)
                    Image.fromarray(crop_resized).save(img_path)
                    counts[split][label] += 1

    print("\n=== Dataset Extraction & Split Summary ===")
    for s in splits:
        tot = counts[s]['real'] + counts[s]['fake']
        print(f"  Split [{s.upper()}]: Total={tot} (Real={counts[s]['real']}, Fake={counts[s]['fake']})")
        
    return counts


if __name__ == "__main__":
    prepare_dataset()
