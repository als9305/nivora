import os
import sys
import glob
import cv2
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "backend")

from models.face_liveness import MultiDomainFeatureExtractor
from scripts.benchmark_face_liveness import extract_video_frames

def main():
    categories = [
        ("GENUINE LIVE", sorted(glob.glob("datasets/genuine/*.mp4"))),
        ("PRINTED PHOTO", sorted(glob.glob("datasets/printed_photo/*.mp4"))),
        ("SCREEN PHOTO", sorted(glob.glob("datasets/screen_photo/*.mp4"))),
        ("SCREEN VIDEO REPLAY", sorted(glob.glob("datasets/screen_video/*.mp4")))
    ]
    
    print("=================================================================")
    print("INDEPENDENT DATASET PHYSICAL FEATURE ANALYSIS")
    print("=================================================================")
    
    for cat_name, v_list in categories:
        print(f"\n--- {cat_name} ({len(v_list)} videos) ---")
        for vp in v_list:
            v_name = os.path.basename(vp)
            frames = extract_video_frames(vp, max_frames=15)
            if not frames:
                continue
                
            y_vars, cr_vars, cb_vars, sat_vars = [], [], [], []
            fft_ratios, lbp_entropies = [], []
            mean_brights = []
            
            # Compute motion delta
            diffs = []
            for i in range(1, len(frames)):
                f1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY) if frames[i-1].ndim == 3 else frames[i-1]
                f2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY) if frames[i].ndim == 3 else frames[i]
                diffs.append(float(np.mean(cv2.absdiff(f1, f2))))
            motion_delta = np.mean(diffs) if diffs else 0.0
            
            for f in frames:
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
                fft_ratios.append(MultiDomainFeatureExtractor.extract_fft_moire_ratio(gray))
                
                lbp_h = MultiDomainFeatureExtractor.extract_lbp_histogram(gray)
                lbp_p = lbp_h[lbp_h > 0]
                lbp_entropies.append(-float(np.sum(lbp_p * np.log2(lbp_p))) if len(lbp_p) > 0 else 0.0)
                
                chroma = MultiDomainFeatureExtractor.extract_chroma_variance(f)
                y_vars.append(chroma["y_var"])
                cr_vars.append(chroma["cr_var"])
                cb_vars.append(chroma["cb_var"])
                sat_vars.append(chroma["sat_var"])
                mean_brights.append(float(np.mean(gray)))
                
            flicker_var = float(np.var(mean_brights)) if len(mean_brights) > 1 else 0.0
            
            print(f"{v_name:<25} | Motion: {motion_delta:<5.2f} | FFT: {np.mean(fft_ratios):<6.3f} | Cr_Var: {np.mean(cr_vars):<5.1f} | Flicker: {flicker_var:<5.1f} | LBP: {np.mean(lbp_entropies):<5.2f}")

if __name__ == "__main__":
    main()
