"""
Blockwise Masking Strategy for Masked Image Modeling (MIM)

Implements blockwise masking as described in the research paper:
- Random block-shaped masks applied to patch tokens
- Used during self-supervised pretraining
- Student receives masked views, teacher receives unmasked views
"""
from __future__ import annotations

import math
import random
from typing import Tuple

import torch


def generate_block_mask(
    grid_size: int = 16,
    mask_ratio: float = 0.4,
    min_block_size: int = 2,
    max_block_size: int = 6,
) -> torch.Tensor:
    """
    Generate a blockwise mask for patch tokens.

    Randomly places rectangular blocks on the patch grid until approximately
    mask_ratio of the patches are masked.

    Args:
        grid_size: Number of patches per side (16 for 256 patches)
        mask_ratio: Target fraction of masked patches (~0.4)
        min_block_size: Minimum block width/height
        max_block_size: Maximum block width/height

    Returns:
        mask: (N,) boolean tensor where True = masked.
              N = grid_size^2 = 256.
    """
    num_patches = grid_size * grid_size
    target_masked = int(num_patches * mask_ratio)

    mask_2d = torch.zeros(grid_size, grid_size, dtype=torch.bool)
    masked_count = 0

    max_attempts = 50
    attempts = 0

    while masked_count < target_masked and attempts < max_attempts:
        # Random block dimensions
        bh = random.randint(min_block_size, min(max_block_size, grid_size))
        bw = random.randint(min_block_size, min(max_block_size, grid_size))

        # Random top-left corner
        top = random.randint(0, grid_size - bh)
        left = random.randint(0, grid_size - bw)

        # Apply block mask
        mask_2d[top:top + bh, left:left + bw] = True
        masked_count = mask_2d.sum().item()
        attempts += 1

    return mask_2d.flatten()  # (N,)


def generate_batch_masks(
    batch_size: int,
    grid_size: int = 16,
    mask_ratio: float = 0.4,
    min_block_size: int = 2,
    max_block_size: int = 6,
) -> torch.Tensor:
    """
    Generate independent block masks for a batch.

    Returns:
        masks: (B, N) boolean tensor
    """
    masks = torch.stack([
        generate_block_mask(grid_size, mask_ratio, min_block_size, max_block_size)
        for _ in range(batch_size)
    ])
    return masks
