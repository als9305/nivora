"""
Module 3 — Face Liveness & Anti-Spoofing Model Engine.
Combines Passive RGB Convolutional Network (PassiveAntiSpoofNet), Multi-Domain Texture/Frequency Descriptors,
and Temporal Micro-Motion Aggregation for presentation attack detection (PAD).
"""

import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional


class PassiveAntiSpoofNet(nn.Module):
    """
    ResNet-style Convolutional Neural Network for Passive RGB Face Anti-Spoofing.
    Evaluates spatial micro-textures, color space distortion, and screen/print artifacts on face crops.
    Includes auxiliary physical feature injection head (LBP + FFT Moiré + Chroma).
    """
    def __init__(self, in_channels: int = 3, aux_dim: int = 8, num_classes: int = 2):
        super().__init__()
        
        # Stem Layer: High-resolution texture feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Residual Block 1: Fine-grained spatial texture features
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        
        # Downsample Block 2: Color space and specular reflectance features
        self.layer2_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False), # 56x56
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.layer2_skip = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )
        
        # Downsample Block 3: High-frequency frequency-domain features
        self.layer3_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False), # 28x28
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        self.layer3_skip = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        )

        # Global Pooling and Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier Head taking CNN features (256) + auxiliary physical features (aux_dim)
        self.fc = nn.Sequential(
            nn.Linear(256 + aux_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, aux_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.stem(x)
        out = F.relu(out + self.layer1(out), inplace=True)
        out = F.relu(self.layer2_skip(out) + self.layer2_conv(out), inplace=True)
        out = F.relu(self.layer3_skip(out) + self.layer3_conv(out), inplace=True)
        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        
        if aux_feats is None:
            aux_feats = torch.zeros((x.size(0), 8), device=x.device)
            
        fused = torch.cat([out, aux_feats], dim=1)
        logits = self.fc(fused)
        return logits


class MultiDomainFeatureExtractor:
    """
    Extracts physical texture, frequency domain, and color space presentation attack descriptors.
    """
    @staticmethod
    def extract_lbp_histogram(gray_crop: np.ndarray, num_points: int = 8, radius: int = 1) -> np.ndarray:
        """Compute 256-bin Local Binary Pattern (LBP) texture histogram."""
        h, w = gray_crop.shape
        if h < 16 or w < 16:
            return np.zeros(256, dtype=np.float32)
            
        padded = np.pad(gray_crop, radius, mode='edge')
        center = padded[radius:-radius, radius:-radius]
        lbp = np.zeros_like(center, dtype=np.uint8)
        
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, 1),           (1, 1),
            (1, 0),   (1, -1),  (0, -1)
        ]
        
        for i, (dy, dx) in enumerate(offsets):
            neighbor = padded[radius+dy : radius+dy+h, radius+dx : radius+dx+w]
            lbp |= ((neighbor >= center).astype(np.uint8) << i)
            
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        norm = np.sum(hist) + 1e-7
        return hist / norm

    @staticmethod
    def extract_fft_moire_ratio(gray_crop: np.ndarray) -> float:
        """Computes High-Frequency Moiré Spectrum Ratio in 2D Fourier Domain."""
        h, w = gray_crop.shape
        if h < 32 or w < 32:
            return 0.0
            
        f = np.fft.fft2(gray_crop.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift) + 1e-7
        
        cy, cx = h // 2, w // 2
        r_inner = min(h, w) // 8
        r_outer = min(h, w) // 2
        
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
        
        low_freq_energy = np.sum(magnitude_spectrum[dist_from_center <= r_inner])
        high_freq_energy = np.sum(magnitude_spectrum[(dist_from_center > r_inner) & (dist_from_center <= r_outer)])
        
        ratio = float(high_freq_energy / (low_freq_energy + 1e-7))
        return round(ratio, 4)

    @staticmethod
    def extract_chroma_variance(bgr_crop: np.ndarray) -> Dict[str, float]:
        """Computes YCbCr and HSV color channel variance."""
        if bgr_crop is None or bgr_crop.size == 0:
            return {"y_var": 0.0, "cb_var": 0.0, "cr_var": 0.0, "sat_var": 0.0}
            
        ycbcr = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
        
        y_var = float(np.var(ycbcr[:, :, 0]))
        cr_var = float(np.var(ycbcr[:, :, 1]))
        cb_var = float(np.var(ycbcr[:, :, 2]))
        sat_var = float(np.var(hsv[:, :, 1]))
        
        return {
            "y_var": round(y_var, 2),
            "cr_var": round(cr_var, 2),
            "cb_var": round(cb_var, 2),
            "sat_var": round(sat_var, 2)
        }

    @staticmethod
    def extract_auxiliary_vector(bgr_crop: np.ndarray) -> np.ndarray:
        """
        Extracts normalized 8-element physical descriptor vector:
        [LBP_Entropy, LBP_Mean, LBP_Std, FFT_Ratio, Y_Var, Cr_Var, Cb_Var, Sat_Var]
        """
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY) if bgr_crop.ndim == 3 else bgr_crop
        lbp_h = MultiDomainFeatureExtractor.extract_lbp_histogram(gray)
        lbp_p = lbp_h[lbp_h > 0]
        entropy = -float(np.sum(lbp_p * np.log2(lbp_p))) if len(lbp_p) > 0 else 0.0
        
        fft_ratio = MultiDomainFeatureExtractor.extract_fft_moire_ratio(gray)
        chroma = MultiDomainFeatureExtractor.extract_chroma_variance(bgr_crop)
        
        vec = np.array([
            entropy / 8.0,
            float(np.mean(lbp_h)),
            float(np.std(lbp_h)),
            min(2.0, fft_ratio),
            min(1.0, chroma["y_var"] / 2500.0),
            min(1.0, chroma["cr_var"] / 1000.0),
            min(1.0, chroma["cb_var"] / 1000.0),
            min(1.0, chroma["sat_var"] / 2500.0)
        ], dtype=np.float32)
        return vec


class TemporalLivenessAggregator:
    """
    Aggregates per-frame anti-spoof evidence over a frame sequence.
    Calculates optical micro-motion to detect static photo presentation attacks.
    """
    def __init__(self, min_motion_delta: float = 0.20, decision_threshold: float = 0.5000):
        self.min_motion_delta = min_motion_delta
        self.decision_threshold = decision_threshold

    def compute_frame_motion_delta(self, frames: List[np.ndarray]) -> float:
        """Calculates average pixel-level frame difference across consecutive grayscale frames."""
        if len(frames) < 2:
            return 1.0  # Neutral fallback for single static frame
            
        diffs = []
        for i in range(1, len(frames)):
            f1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY) if frames[i-1].ndim == 3 else frames[i-1]
            f2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY) if frames[i].ndim == 3 else frames[i]
            
            if f1.shape != f2.shape:
                f2 = cv2.resize(f2, (f1.shape[1], f1.shape[0]))
                
            diff = float(np.mean(cv2.absdiff(f1, f2)))
            diffs.append(diff)
            
        return round(float(np.mean(diffs)), 3)

    def aggregate(
        self,
        frame_scores: List[float],
        motion_delta: float,
        quality_passed: bool = True,
        fft_ratio: float = 0.0,
        flicker_var: float = 0.0,
        lbp_entropy: float = 0.0
    ) -> Dict[str, Any]:
        """Aggregates frame scores into unified liveness verdict using multi-domain physical guards and decision threshold tau*."""
        if not frame_scores:
            return {
                "liveness": "INSUFFICIENT_EVIDENCE",
                "score": 0.0,
                "confidence": 0.0,
                "attack_type": None,
                "reason": "No valid frame scores extracted."
            }
            
        # 1. Micro-motion static photo detection guard
        if len(frame_scores) > 1 and motion_delta < self.min_motion_delta:
            return {
                "liveness": "SPOOF",
                "score": round(min(frame_scores) * 0.2, 4),
                "confidence": 0.95,
                "attack_type": "PRINTED_PHOTO_OR_STATIC_REPLAY",
                "reason": f"Static photo presentation attack detected. Frame motion delta {motion_delta} < noise threshold {self.min_motion_delta}."
            }
            
        # 2. Digital Screen Video Replay presentation attack guard
        if fft_ratio >= 1.65 and flicker_var >= 40.0:
            return {
                "liveness": "SPOOF",
                "score": round(min(frame_scores) * 0.25, 4),
                "confidence": 0.92,
                "attack_type": "SCREEN_VIDEO_REPLAY",
                "reason": f"Digital screen video replay presentation attack detected. Moiré grid spectrum ratio {fft_ratio:.3f} >= 1.65 and screen refresh flicker {flicker_var:.1f} >= 40.0."
            }
            
        # 3. Trimmed Mean Score Aggregation
        sorted_scores = sorted(frame_scores)
        if len(sorted_scores) >= 5:
            trim_k = max(1, len(sorted_scores) // 10)
            trimmed = sorted_scores[trim_k:-trim_k]
        else:
            trimmed = sorted_scores
            
        avg_score = float(np.mean(trimmed))
        
        # 4. Decision Rule
        if avg_score >= self.decision_threshold:
            liveness_status = "LIVE"
            attack_type = None
            reason = f"Bona fide live human presentation verified. Passive score {round(avg_score, 4)} >= threshold {self.decision_threshold}."
        else:
            liveness_status = "SPOOF"
            attack_type = "PRESENTATION_ATTACK_REPLAY_OR_PRINT"
            reason = f"Passive anti-spoof score {round(avg_score, 4)} below threshold {self.decision_threshold}."
            
        return {
            "liveness": liveness_status,
            "score": round(avg_score, 4),
            "confidence": round(abs(avg_score - 0.5) * 2.0, 4),
            "attack_type": attack_type,
            "reason": reason
        }


def load_passive_liveness_model(device: str = "cpu") -> PassiveAntiSpoofNet:
    model = PassiveAntiSpoofNet(in_channels=3, aux_dim=8, num_classes=2)
    model.eval()
    return model.to(device)
