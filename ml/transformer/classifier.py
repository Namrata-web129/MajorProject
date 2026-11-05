"""
Graph Transformer Classifier for Deepfake Detection

Research architecture:
- 3 Transformer blocks
- 8 attention heads
- 256 embedding dimension
- 512 MLP dimension
- LayerNorm + residual connections
- CLS token + classification head → Real / Fake

Input: GCN output features (B, N, D)
Output: Real/Fake prediction with probabilities
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GraphTransformerBlock(nn.Module):
    """
    Single Transformer block for the Graph Transformer.

    Implements:
        Q = X W_Q,  K = X W_K,  V = X W_V
        Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V
        MultiHead = Concat(head_1, ..., head_h) W_O
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        # Multi-head self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

        # MLP
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Store attention weights for explainability
        self.attn_weights: Optional[torch.Tensor] = None
        self.capture_attention_gradients = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N+1, D) — N graph node tokens + 1 CLS token

        Returns:
            out: (B, N+1, D)
        """
        B, L, D = x.shape

        # Pre-norm + Multi-head self-attention
        x_norm = self.norm1(x)

        Q = self.q_proj(x_norm)  # (B, L, D)
        K = self.k_proj(x_norm)
        V = self.v_proj(x_norm)

        # Reshape for multi-head: (B, num_heads, L, head_dim)
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** 0.5
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, L, L)
        attn = F.softmax(attn, dim=-1)
        if self.capture_attention_gradients and torch.is_grad_enabled():
            attn.retain_grad()
            self.attn_weights = attn
        else:
            self.attn_weights = attn.detach()
        attn = self.attn_dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, V)  # (B, H, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)

        # Residual
        x = x + out

        # Pre-norm + MLP + Residual
        x = x + self.mlp(self.norm2(x))

        return x


class GraphTransformerClassifier(nn.Module):
    """
    Graph Transformer Classifier.

    Takes GCN-processed node features, prepends a learnable CLS token,
    adds positional embeddings, processes through Transformer blocks,
    and classifies as Real (0) or Fake (1).

    Returns:
        - Predicted class (0 = REAL, 1 = FAKE)
        - Real probability
        - Fake probability
        - Raw logits
        - Feature embeddings
    """

    def __init__(
        self,
        num_nodes: int = 256,
        embed_dim: int = 256,
        num_blocks: int = 3,
        num_heads: int = 8,
        mlp_dim: int = 512,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks

        # Learnable CLS token for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding for graph nodes + CLS
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_nodes + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            GraphTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        # Final norm
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
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

    def forward(
        self,
        node_features: torch.Tensor,  # (B, N, D) — GCN output
    ) -> dict:
        """
        Forward pass through the Graph Transformer classifier.

        Args:
            node_features: (B, N, D) from GCN

        Returns:
            dict with:
                'logits': (B, 2) raw logits
                'probabilities': (B, 2) softmax probabilities
                'prediction': (B,) — 0=REAL, 1=FAKE
                'real_probability': (B,)
                'fake_probability': (B,)
                'cls_embedding': (B, D)
                'node_embeddings': (B, N, D) — final node representations
        """
        B, N, D = node_features.shape

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, node_features], dim=1)  # (B, N+1, D)

        # Add positional embeddings
        x = x + self.pos_embed[:, :N + 1, :]
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm(x)

        # CLS token → classification
        cls_out = x[:, 0]         # (B, D)
        node_out = x[:, 1:]       # (B, N, D)

        logits = self.head(cls_out)  # (B, 2)
        probs = F.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "probabilities": probs,
            "prediction": torch.argmax(probs, dim=-1),
            "real_probability": probs[:, 0],
            "fake_probability": probs[:, 1],
            "cls_embedding": cls_out,
            "node_embeddings": node_out,
        }

    def get_attention_maps(self) -> list:
        """
        Retrieve attention weights from all Transformer blocks.
        Returns list of (B, num_heads, N+1, N+1) tensors.
        """
        maps = []
        for block in self.blocks:
            if block.attn_weights is not None:
                maps.append(block.attn_weights)
        return maps

    def set_attention_gradient_capture(self, enabled: bool) -> None:
        """Enable gradient retention for an explainability forward pass."""
        for block in self.blocks:
            block.capture_attention_gradients = enabled
