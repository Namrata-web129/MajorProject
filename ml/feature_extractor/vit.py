"""
Vision Transformer (ViT) Feature Extractor
Implementation following the research paper specifications:
- Input: 320×320 image
- Patch size: 20×20
- Total patches: 256
- CLS token included
- Embedding dimension: 256

This ViT serves as the backbone feature extractor, trained via
self-supervised student-teacher learning with Masked Image Modeling.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PatchEmbedding(nn.Module):
    """
    Project image patches to embedding vectors using a convolutional layer.
    This is equivalent to extracting non-overlapping patches and linearly projecting them.
    """

    def __init__(
        self,
        image_size: int = 320,
        patch_size: int = 20,
        in_channels: int = 3,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size  # 16
        self.num_patches = self.grid_size ** 2     # 256

        # Conv2D with kernel_size = stride = patch_size acts as patch extraction + linear projection
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor

        Returns:
            patch_tokens: (B, N, D) where N = num_patches, D = embed_dim
        """
        B, C, H, W = x.shape
        assert H == self.image_size and W == self.image_size, (
            f"Expected ({self.image_size}, {self.image_size}), got ({H}, {W})"
        )
        # (B, D, grid_h, grid_w)
        x = self.projection(x)
        # Flatten spatial dims: (B, D, N) → (B, N, D)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    """
    Standard Transformer encoder block with:
    - Multi-head self-attention
    - LayerNorm (pre-norm style)
    - MLP with GELU activation
    - Residual connections
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads,
            dropout=attn_dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

        # Store attention weights for explainability
        self.attn_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N+1, D)  (N patch tokens + 1 CLS token)

        Returns:
            out: (B, N+1, D)
        """
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm, need_weights=True)
        self.attn_weights = attn_weights  # (B, N+1, N+1)
        x = x + attn_out

        # MLP with pre-norm
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer feature extractor.

    Architecture:
    - Patch embedding (Conv2D)
    - Learnable CLS token
    - Positional embeddings
    - Transformer encoder blocks
    - Final LayerNorm
    - Output: CLS token embedding + patch token embeddings

    Used for both student and teacher networks in self-supervised learning.
    """

    def __init__(
        self,
        image_size: int = 320,
        patch_size: int = 20,
        in_channels: int = 3,
        embed_dim: int = 256,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches  # 256

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings (learnable)
        # +1 for CLS token → 257 positions
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(depth)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Projection heads for self-supervised learning
        # CLS head: projects CLS token to distribution for self-distillation
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # Patch head: projects patch tokens to distribution for MIM
        self.patch_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise weights using truncated normal (ViT convention)."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass.

        Args:
            x: (B, C, H, W) input image tensor
            mask: (B, N) boolean mask — True = masked (for MIM).
                  Masked patches are replaced with a learnable mask token
                  or zeroed out.

        Returns:
            dict with keys:
                'cls_token': (B, D)  — CLS embedding
                'patch_tokens': (B, N, D) — patch embeddings
                'cls_head_out': (B, D) — CLS projection head output
                'patch_head_out': (B, N, D) — patch projection head outputs
        """
        B = x.shape[0]

        # Patch embedding: (B, N, D)
        patch_tokens = self.patch_embed(x)
        N = patch_tokens.shape[1]

        # Apply masking (zero out masked patches, as in iBOT/DINO)
        if mask is not None:
            # mask: (B, N) bool — True = masked
            mask_expanded = mask.unsqueeze(-1).float()  # (B, N, 1)
            patch_tokens = patch_tokens * (1.0 - mask_expanded)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)  # (B, N+1, D)

        # Add positional embeddings
        tokens = tokens + self.pos_embed[:, :N + 1, :]
        tokens = self.pos_drop(tokens)

        # Transformer blocks
        for block in self.blocks:
            tokens = block(tokens)

        # Final norm
        tokens = self.norm(tokens)

        # Split CLS and patch tokens
        cls_out = tokens[:, 0]         # (B, D)
        patch_out = tokens[:, 1:]      # (B, N, D)

        # Projection heads
        cls_head_out = self.cls_head(cls_out)         # (B, D)
        patch_head_out = self.patch_head(patch_out)   # (B, N, D)

        return {
            "cls_token": cls_out,
            "patch_tokens": patch_out,
            "cls_head_out": cls_head_out,
            "patch_head_out": patch_head_out,
        }

    def get_attention_maps(self) -> list:
        """
        Retrieve attention weights from all transformer blocks.
        Used for explainability / relevancy maps.

        Returns:
            List of (B, num_heads, N+1, N+1) attention weight tensors.
        """
        maps = []
        for block in self.blocks:
            if block.attn_weights is not None:
                maps.append(block.attn_weights)
        return maps

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method: extract CLS and patch features without heads.

        Returns:
            cls_features: (B, D)
            patch_features: (B, N, D)
        """
        out = self.forward(x)
        return out["cls_token"], out["patch_tokens"]
