"""
Heatmap Generation and Visualization

Converts node-level relevance scores into spatial heatmaps
and generates overlay visualizations on face images.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def relevance_to_heatmap(
    relevance: np.ndarray,
    grid_size: int = 16,
    image_size: int = 320,
) -> np.ndarray:
    """
    Convert per-node relevance scores to a spatial heatmap.

    Args:
        relevance: (N,) relevance scores in [0, 1] for each patch
        grid_size: Patch grid size (16)
        image_size: Target image size for the heatmap (320)

    Returns:
        heatmap: (image_size, image_size) float array in [0, 1]
    """
    N = grid_size * grid_size
    assert len(relevance) >= N, f"Expected {N} relevance scores, got {len(relevance)}"

    # Reshape to grid
    relevance_grid = relevance[:N].reshape(grid_size, grid_size)

    # Upscale to image size using bicubic interpolation
    heatmap = cv2.resize(
        relevance_grid.astype(np.float32),
        (image_size, image_size),
        interpolation=cv2.INTER_CUBIC,
    )

    # Clip and normalise
    heatmap = np.clip(heatmap, 0.0, 1.0)

    return heatmap


def generate_heatmap_overlay(
    face_image: np.ndarray,
    relevance: np.ndarray,
    output_path: str,
    grid_size: int = 16,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.5,
) -> str:
    """
    Generate a heatmap overlay on a face image and save to disk.

    Args:
        face_image: (H, W, 3) BGR face image
        relevance: (N,) relevance scores
        output_path: Path to save the overlay image
        grid_size: Patch grid size
        colormap: OpenCV colormap for visualization
        alpha: Blending alpha (0 = face only, 1 = heatmap only)

    Returns:
        output_path: Path to saved image
    """
    h, w = face_image.shape[:2]

    # Generate heatmap at face resolution
    heatmap = relevance_to_heatmap(relevance, grid_size, max(h, w))
    heatmap = cv2.resize(heatmap, (w, h))

    # Convert to colour-mapped image (0-255 uint8)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

    # Blend with face image
    overlay = cv2.addWeighted(face_image, 1.0 - alpha, heatmap_color, alpha, 0)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, overlay)

    return output_path


def generate_patch_visualization(
    face_image: np.ndarray,
    relevance: np.ndarray,
    output_path: str,
    grid_size: int = 16,
    top_k: int = 20,
) -> str:
    """
    Generate a patch-level visualization showing the most relevant patches.

    Draws rectangles around the top-K most relevant patches.

    Args:
        face_image: (H, W, 3) BGR face image
        relevance: (N,) relevance scores
        output_path: Path to save
        grid_size: Patch grid size
        top_k: Number of top patches to highlight

    Returns:
        output_path
    """
    h, w = face_image.shape[:2]
    patch_h = h // grid_size
    patch_w = w // grid_size

    vis = face_image.copy()

    # Sort patches by relevance (descending)
    N = grid_size * grid_size
    indices = np.argsort(relevance[:N])[::-1][:top_k]

    for rank, idx in enumerate(indices):
        row = idx // grid_size
        col = idx % grid_size

        x1 = col * patch_w
        y1 = row * patch_h
        x2 = x1 + patch_w
        y2 = y1 + patch_h

        # Colour: red for most relevant, fading to yellow
        intensity = 1.0 - (rank / top_k)
        color = (0, int(255 * (1 - intensity)), 255)  # BGR: red → yellow

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, vis)

    return output_path


def generate_standalone_heatmap(
    relevance: np.ndarray,
    output_path: str,
    grid_size: int = 16,
    size: int = 320,
    colormap: int = cv2.COLORMAP_JET,
) -> str:
    """Generate a standalone heatmap image (no face overlay)."""
    heatmap = relevance_to_heatmap(relevance, grid_size, size)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, heatmap_color)
    return output_path
