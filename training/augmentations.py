"""
Input Preprocessing and Data Augmentation Pipeline for DeepQShield.
Implements 224x224 input normalization and research paper augmentations.
"""

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from typing import Union, Tuple, Dict, Any

# Standard ImageNet / Pretrained Normalization Constants
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_inference_transform(img_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Deterministic inference preprocessing pipeline.
    Resize -> Center Crop -> ToTensor -> Normalize.
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


def get_validation_transform(img_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Validation preprocessing pipeline.
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])


def get_train_transform(img_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Training data augmentation pipeline based on DeepQShield paper.
    Includes flip, rotation, color jitter (brightness/contrast/hue), and RandomErasing (coarse dropout).
    """
    return transforms.Compose([
        transforms.Resize((int(img_size[0] * 1.14), int(img_size[1] * 1.14))),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), value=0)
    ])


def preprocess_image_array(
    image_np: np.ndarray,
    img_size: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """
    Preprocess BGR/RGB numpy array into a normalized PyTorch tensor (1, 3, 224, 224).
    
    Args:
        image_np: BGR or RGB numpy image array
        img_size: Target (H, W) tuple
        
    Returns:
        Tensor of shape (1, 3, 224, 224) ready for model inference
    """
    if not isinstance(image_np, np.ndarray):
        raise TypeError(f"Expected numpy ndarray, got {type(image_np)}")
        
    # Convert BGR to RGB if needed
    if image_np.ndim == 3 and image_np.shape[2] == 3:
        image_rgb = image_np[:, :, ::-1] # BGR to RGB
    else:
        image_rgb = image_np
        
    pil_img = Image.fromarray(image_rgb.astype('uint8'))
    transform = get_inference_transform(img_size)
    tensor = transform(pil_img)
    return tensor.unsqueeze(0)
