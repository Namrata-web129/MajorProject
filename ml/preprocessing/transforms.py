"""
Preprocessing Module
Image normalization, augmentation, and preparation for the ViT pipeline.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)

# ImageNet-derived normalization values (standard for ViT models)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk.  Returns BGR numpy array.
    Raises FileNotFoundError / ValueError on failure.
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    return img


def resize_face(image: np.ndarray, size: Tuple[int, int] = (320, 320)) -> np.ndarray:
    """Resize a face crop to (width, height). Returns BGR numpy array."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV default) to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def normalize_tensor(
    tensor: torch.Tensor,
    mean: Tuple[float, ...] = IMAGENET_MEAN,
    std: Tuple[float, ...] = IMAGENET_STD,
) -> torch.Tensor:
    """
    Normalize a [C, H, W] float tensor using channel-wise mean/std.
    """
    transform = transforms.Normalize(mean=mean, std=std)
    return transform(tensor)


def image_to_tensor(
    image: np.ndarray,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Convert a BGR numpy image (H, W, C) to a normalised float tensor (C, H, W).
    """
    rgb = bgr_to_rgb(image)
    # HWC uint8 → CHW float [0, 1]
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    if normalize:
        tensor = normalize_tensor(tensor)
    return tensor


def prepare_face_for_model(
    face_bgr: np.ndarray,
    image_size: int = 320,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Full preprocessing: resize → to_tensor → normalise.
    Returns a (C, H, W) float tensor ready for batch unsqueeze.
    """
    resized = resize_face(face_bgr, (image_size, image_size))
    return image_to_tensor(resized, normalize=normalize)


# ── Augmentation for self-supervised pretraining ──────────────────────────────

def get_ssl_augmentation(image_size: int = 320) -> transforms.Compose:
    """
    Two-view augmentation pipeline for self-supervised pretraining.
    Follows DINO/iBOT-style augmentations: random crop, flip, colour jitter,
    Gaussian blur, solarisation.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1,
            ),
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        ], p=0.5),
        transforms.RandomSolarize(threshold=128, p=0.2),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform(image_size: int = 320) -> transforms.Compose:
    """Deterministic evaluation transform: resize + centre-crop + normalise."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
