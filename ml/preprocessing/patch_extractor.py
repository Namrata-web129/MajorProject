"""
Patch Extractor
Extracts non-overlapping patches from a face image for the Vision Transformer.
Research configuration: 320×320 image → 20×20 patches → 256 patch tokens.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class PatchExtractor:
    """
    Extract non-overlapping patches from an image tensor.

    For a 320×320 image with patch_size=20:
        grid_h = grid_w = 320 / 20 = 16
        total patches = 16 × 16 = 256
    """

    def __init__(self, image_size: int = 320, patch_size: int = 20) -> None:
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size  # 16
        self.num_patches = self.grid_size ** 2     # 256

    def extract_patches(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract non-overlapping patches from an image tensor.

        Args:
            image_tensor: (C, H, W) or (B, C, H, W)

        Returns:
            patches: (N, C, patch_size, patch_size) or (B, N, C, patch_size, patch_size)
                     where N = num_patches
        """
        batched = image_tensor.dim() == 4
        if not batched:
            image_tensor = image_tensor.unsqueeze(0)

        B, C, H, W = image_tensor.shape
        assert H == self.image_size and W == self.image_size, (
            f"Expected ({self.image_size}, {self.image_size}), got ({H}, {W})"
        )

        ps = self.patch_size
        gs = self.grid_size

        # Reshape: (B, C, gs, ps, gs, ps) → (B, gs*gs, C, ps, ps)
        patches = image_tensor.reshape(B, C, gs, ps, gs, ps)
        patches = patches.permute(0, 2, 4, 1, 3, 5)  # (B, gs, gs, C, ps, ps)
        patches = patches.reshape(B, gs * gs, C, ps, ps)  # (B, N, C, ps, ps)

        if not batched:
            patches = patches.squeeze(0)  # (N, C, ps, ps)

        return patches

    def flatten_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Flatten each patch into a vector.

        Args:
            patches: (B, N, C, ps, ps) or (N, C, ps, ps)

        Returns:
            flat: (B, N, C*ps*ps) or (N, C*ps*ps)
        """
        if patches.dim() == 4:
            N, C, ps1, ps2 = patches.shape
            return patches.reshape(N, C * ps1 * ps2)
        else:
            B, N, C, ps1, ps2 = patches.shape
            return patches.reshape(B, N, C * ps1 * ps2)

    def get_patch_coords(self) -> List[Tuple[int, int, int, int]]:
        """
        Return the (x, y, w, h) coordinates for each patch in image space.
        Ordered from top-left to bottom-right, row by row.
        """
        coords = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x = col * self.patch_size
                y = row * self.patch_size
                coords.append((x, y, self.patch_size, self.patch_size))
        return coords

    def patches_to_image(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct an image from patches (inverse of extract_patches).

        Args:
            patches: (N, C, ps, ps) or (B, N, C, ps, ps)

        Returns:
            image: (C, H, W) or (B, C, H, W)
        """
        batched = patches.dim() == 5
        if not batched:
            patches = patches.unsqueeze(0)

        B, N, C, ps, _ = patches.shape
        gs = self.grid_size
        assert N == gs * gs

        patches = patches.reshape(B, gs, gs, C, ps, ps)
        patches = patches.permute(0, 3, 1, 4, 2, 5)  # (B, C, gs, ps, gs, ps)
        image = patches.reshape(B, C, gs * ps, gs * ps)

        if not batched:
            image = image.squeeze(0)
        return image
