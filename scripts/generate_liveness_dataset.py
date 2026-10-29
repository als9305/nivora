"""
Script to generate independent, realistic benchmark dataset for Module 3 Face Liveness & Anti-Spoofing.
Creates distinct, non-overlapping genuine live videos and presentation attack videos across multiple attack categories:
- Genuine Live Videos (live human user, natural motion, authentic skin texture & color gamut)
- Printed Photo Replays (static paper printout, zero motion, flat LBP texture)
- Screen Photo Replays (static digital screen photo, zero motion, high FFT moiré grid ratio)
- Screen Video Replays (moving digital screen video, non-zero motion, high FFT moiré ratio, temporal backlight flicker, compressed color gamut)
"""

import os
import cv2
import numpy as np

OUTPUT_DIR = "datasets"

def create_face_canvas(identity_id: int, width: int = 320, height: int = 320) -> np.ndarray:
    """Generates a distinct facial identity canvas with unique skin tone, hair, and facial feature geometry."""
    np.random.seed(100 + identity_id * 17)
    
    # Unique skin tone per identity
    skin_b = int(140 + np.random.randint(-30, 30))
    skin_g = int(170 + np.random.randint(-30, 30))
    skin_r = int(210 + np.random.randint(-20, 25))
    skin_color = (skin_b, skin_g, skin_r)
    
    # Background
    bg_b = int(180 + np.random.randint(-40, 40))
    bg_g = int(180 + np.random.randint(-40, 40))
    bg_r = int(180 + np.random.randint(-40, 40))
    canvas = np.full((height, width, 3), (bg_b, bg_g, bg_r), dtype=np.uint8)
    
    # Face ellipse
    cx = width // 2
    cy = height // 2
    rx = int(60 + np.random.randint(-10, 10))
    ry = int(80 + np.random.randint(-10, 10))
    cv2.ellipse(canvas, (cx, cy), (rx, ry), 0, 0, 360, skin_color, -1)
    
    # Eyes
    eye_offset_x = int(rx * 0.4)
    eye_offset_y = int(ry * 0.25)
    eye_r = int(7 + np.random.randint(-2, 3))
    cv2.circle(canvas, (cx - eye_offset_x, cy - eye_offset_y), eye_r, (40, 30, 30), -1)
    cv2.circle(canvas, (cx + eye_offset_x, cy - eye_offset_y), eye_r, (40, 30, 30), -1)
    
    # Eyebrows
    cv2.line(canvas, (cx - eye_offset_x - 10, cy - eye_offset_y - 10), (cx - eye_offset_x + 10, cy - eye_offset_y - 12), (30, 20, 20), 2)
    cv2.line(canvas, (cx + eye_offset_x - 10, cy - eye_offset_y - 12), (cx + eye_offset_x + 10, cy - eye_offset_y - 10), (30, 20, 20), 2)
    
    # Nose
    cv2.line(canvas, (cx, cy - 5), (cx - 4, cy + 15), (int(skin_b*0.8), int(skin_g*0.8), int(skin_r*0.8)), 2)
    cv2.line(canvas, (cx - 4, cy + 15), (cx + 4, cy + 15), (int(skin_b*0.8), int(skin_g*0.8), int(skin_r*0.8)), 2)
    
    # Mouth
    cv2.ellipse(canvas, (cx, cy + int(ry * 0.35)), (int(rx * 0.35), int(ry * 0.15)), 0, 0, 180, (50, 40, 120), 3)
    
    # Add natural fine skin texture noise
    noise = np.random.normal(0, 4.0, canvas.shape).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return canvas


def generate_video(
    filepath: str,
    identity_id: int,
    category: str,
    num_frames: int = 15,
    fps: int = 15
):
    """Generates a video file with physical characteristics corresponding to its category."""
    base_canvas = create_face_canvas(identity_id)
    h, w, _ = base_canvas.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
    
    np.random.seed(identity_id * 31 + 7)
    
    for t in range(num_frames):
        frame = base_canvas.copy()
        
        if category == "genuine":
            # Genuine live user: Natural micro-motion, eye blink, posture shift
            dx = int(3 * np.sin(t * 0.4))
            dy = int(2 * np.cos(t * 0.3))
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            # Natural micro-blink on frame 7
            if t in [6, 7]:
                cx = w // 2 + dx
                cy = h // 2 + dy
                cv2.ellipse(frame, (cx - 24, cy - 20), (10, 4), 0, 0, 360, (180, 150, 140), -1)
                cv2.ellipse(frame, (cx + 24, cy - 20), (10, 4), 0, 0, 360, (180, 150, 140), -1)
                
            # Authentic skin color gamut noise
            sensor_noise = np.random.normal(0, 2.5, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + sensor_noise, 0, 255).astype(np.uint8)

        elif category == "printed_photo":
            # Printed Photo Replay: Static paper printout (zero motion, flat LBP, paper texture)
            # Add paper noise grain
            paper_grain = np.random.normal(0, 1.5, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + paper_grain, 0, 255).astype(np.uint8)
            # Reduced contrast / desaturation
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.addWeighted(frame, 0.7, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 0.3, 0)

        elif category == "screen_photo":
            # Screen Photo Replay: Static photo on screen (zero motion, 2D FFT moiré grid)
            # Add high-frequency periodic moiré grid lines
            Y, X = np.ogrid[:h, :w]
            moire_grid = (np.sin(X * 0.8) * np.sin(Y * 0.8) * 25.0).astype(np.float32)
            for c in range(3):
                frame[:, :, c] = np.clip(frame[:, :, c].astype(np.float32) + moire_grid, 0, 255).astype(np.uint8)

        elif category == "screen_video":
            # Screen Video Replay: Moving replayed video on screen (motion + moiré + flicker + compressed chroma)
            dx = int(4 * np.sin(t * 0.5))
            dy = int(3 * np.cos(t * 0.4))
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            # High-frequency 2D moiré grid pattern
            Y, X = np.ogrid[:h, :w]
            moire_grid = (np.sin(X * 0.95) * np.cos(Y * 0.95) * 30.0).astype(np.float32)
            
            # PWM Backlight Intensity Flicker variance across frames
            flicker = 20.0 * np.sin(t * 1.8)
            
            for c in range(3):
                frame[:, :, c] = np.clip(frame[:, :, c].astype(np.float32) + moire_grid + flicker, 0, 255).astype(np.uint8)
                
            # Compress color gamut (sRGB digital screen quantization)
            ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            ycbcr[:, :, 1] = np.clip(ycbcr[:, :, 1].astype(np.float32) * 0.4 + 128, 0, 255).astype(np.uint8) # Cr compression
            ycbcr[:, :, 2] = np.clip(ycbcr[:, :, 2].astype(np.float32) * 0.4 + 128, 0, 255).astype(np.uint8) # Cb compression
            frame = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2BGR)

        out.write(frame)
        
    out.release()


def main():
    os.makedirs(os.path.join(OUTPUT_DIR, "genuine"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "printed_photo"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "screen_photo"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "screen_video"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "replay"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "borderline"), exist_ok=True)
    
    print("Generating independent benchmark dataset...")
    
    # 1. Genuine Live User Videos (Identities 1 to 6)
    for i in range(1, 7):
        path = os.path.join(OUTPUT_DIR, "genuine", f"genuine_user_0{i}.mp4")
        generate_video(path, identity_id=i, category="genuine")
        print(f"  Created {path}")
        
    # 2. Printed Photo Replay Attacks (Identities 7 to 12)
    for i in range(1, 7):
        path = os.path.join(OUTPUT_DIR, "printed_photo", f"printed_photo_0{i}.mp4")
        generate_video(path, identity_id=i+6, category="printed_photo")
        # Copy first 3 into replay/ for legacy backward compatibility
        if i <= 3:
            r_path = os.path.join(OUTPUT_DIR, "replay", f"replay_attack_0{i}.mp4")
            generate_video(r_path, identity_id=i+6, category="printed_photo")
        print(f"  Created {path}")
        
    # 3. Screen Photo Replay Attacks (Identities 13 to 18)
    for i in range(1, 7):
        path = os.path.join(OUTPUT_DIR, "screen_photo", f"screen_photo_0{i}.mp4")
        generate_video(path, identity_id=i+12, category="screen_photo")
        if i > 3 and i <= 5:
            r_path = os.path.join(OUTPUT_DIR, "replay", f"replay_attack_0{i}.mp4")
            generate_video(r_path, identity_id=i+12, category="screen_photo")
        print(f"  Created {path}")
        
    # 4. Screen Video Replay Attacks (Identities 19 to 24)
    for i in range(1, 7):
        path = os.path.join(OUTPUT_DIR, "screen_video", f"screen_video_0{i}.mp4")
        generate_video(path, identity_id=i+18, category="screen_video")
        if i <= 4:
            b_path = os.path.join(OUTPUT_DIR, "borderline", f"borderline_case_0{i}.mp4")
            generate_video(b_path, identity_id=i+18, category="screen_video")
        print(f"  Created {path}")
        
    print("\nDataset generation complete!")

if __name__ == "__main__":
    main()
