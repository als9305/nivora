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
    gen_videos = sorted(glob.glob("datasets/genuine/*.mp4"))
    borderline_videos = sorted(glob.glob("datasets/borderline/*.mp4"))
    replay_videos = sorted(glob.glob("datasets/replay/*.mp4"))
    
    print("=================================================================")
    print("PHYSICAL FEATURE ANALYSIS: GENUINE VS BORDERLINE VS REPLAY")
    print("=================================================================")
    
    categories = [
        ("GENUINE LIVE", gen_videos),
        ("BORDERLINE VIDEO REPLAY", borderline_videos),
        ("STATIC PHOTO REPLAY", replay_videos)
    ]
    
    for cat_name, v_list in categories:
        print(f"\n--- {cat_name} ---")
        for vp in v_list:
            v_name = os.path.basename(vp)
            frames = extract_video_frames(vp, max_frames=10)
            if not frames:
                continue
                
            y_vars, cr_vars, cb_vars, sat_vars = [], [], [], []
            fft_ratios, lbp_entropies = [], []
            
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
                
            print(f"{v_name:<25} | Cr_Var: {np.mean(cr_vars):<6.1f} | Cb_Var: {np.mean(cb_vars):<6.1f} | Sat_Var: {np.mean(sat_vars):<7.1f} | FFT: {np.mean(fft_ratios):<6.3f} | LBP_H: {np.mean(lbp_entropies):<5.2f}")

if __name__ == "__main__":
    main()
