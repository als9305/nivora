"""
Module 2 — Face Recognition & Embedding Engine for Nivora.
Uses pre-trained InceptionResnetV1 (VGGFace2) to generate 512-dimensional
L2-normalized feature embeddings for face identity enrollment and verification.
"""

import cv2
import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from facenet_pytorch import InceptionResnetV1


class FaceRecognitionEngine:
    """
    Production Face Recognition & Embedding Engine.
    Generates 512-d L2-normalized feature vectors for face verification.
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.embedding_dim = 512
        self.crop_size = (160, 160)

    def preprocess_crop(self, face_bgr: np.ndarray) -> torch.Tensor:
        """
        Preprocesses a BGR face crop array into normalized PyTorch tensor (1x3x160x160).
        """
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Empty or invalid face crop provided for preprocessing.")

        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        # Resize to fixed crop size 160x160
        face_resized = cv2.resize(face_rgb, self.crop_size, interpolation=cv2.INTER_AREA)

        # Normalize pixel values to [-1.0, 1.0] as expected by InceptionResnetV1
        face_normalized = (face_resized.astype(np.float32) - 127.5) / 128.0

        # Transpose from (H, W, C) to (C, H, W)
        tensor = torch.tensor(face_normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def extract_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Extracts 512-d L2-normalized embedding vector from a BGR face crop.
        Returns a float32 numpy array of shape (512,) with ||e||_2 == 1.0.
        """
        tensor = self.preprocess_crop(face_bgr)
        raw_embedding = self.model(tensor).cpu().numpy().flatten()

        # Compute exact L2 norm
        norm = np.linalg.norm(raw_embedding)
        if norm > 1e-8:
            normalized_embedding = raw_embedding / norm
        else:
            normalized_embedding = raw_embedding

        return normalized_embedding.astype(np.float32)

    @staticmethod
    def compute_cosine_similarity(vec1: List[float] | np.ndarray, vec2: List[float] | np.ndarray) -> float:
        """
        Computes exact Cosine Similarity between two L2-normalized embedding vectors.
        S(a, b) = a . b / (||a||2 * ||b||2)
        Returns score in range [-1.0, 1.0], bounded to [0.0, 1.0].
        """
        if vec1 is None or vec2 is None:
            return 0.0

        v1 = np.array(vec1, dtype=np.float32).flatten()
        v2 = np.array(vec2, dtype=np.float32).flatten()

        if len(v1) == 0 or len(v2) == 0 or len(v1) != len(v2):
            return 0.0

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0

        sim = float(np.dot(v1, v2) / (norm1 * norm2))
        return float(np.clip(sim, 0.0, 1.0))

    @staticmethod
    def aggregate_template_embeddings(embeddings: List[np.ndarray | List[float]]) -> List[float]:
        """
        Aggregates multiple face sample embeddings into a single L2-normalized template centroid.
        e_template = sum(e_i) / ||sum(e_i)||_2
        """
        if not embeddings:
            raise ValueError("No embeddings provided for aggregation.")

        valid_vecs = []
        for e in embeddings:
            arr = np.array(e, dtype=np.float32).flatten()
            if len(arr) == 512 and np.linalg.norm(arr) > 1e-8:
                valid_vecs.append(arr)

        if not valid_vecs:
            raise ValueError("No valid 512-d embeddings available to aggregate.")

        mean_vec = np.mean(valid_vecs, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-8:
            template_vec = mean_vec / norm
        else:
            template_vec = mean_vec

        return template_vec.tolist()


# Global singleton instance for efficient model reuse
face_engine_instance = FaceRecognitionEngine()
