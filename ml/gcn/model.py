"""
Graph Convolutional Network (GCN)

Implements the GCN propagation rule from Kipf & Welling (2017):
    H^(l+1) = ReLU( Â H^(l) W^(l) )

The GCN learns local relationships between facial patches.
Configurable number of layers, embedding dimensions, and dropout.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GCNLayer(nn.Module):
    """
    Single Graph Convolutional Layer.

    Implements: H' = ReLU( Â H W + b )

    Supports both dense adjacency (N, N) and batched mode.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization (standard for GCN)."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,           # (B, N, D_in) or (N, D_in)
        adj_norm: torch.Tensor,     # (N, N) normalised adjacency
    ) -> torch.Tensor:
        """
        GCN propagation: Â X W + b

        Args:
            x: Node features (batched or single)
            adj_norm: Normalised adjacency matrix Â

        Returns:
            out: (B, N, D_out) or (N, D_out) transformed features
        """
        # Linear transform: X W → (*, N, D_out)
        support = torch.matmul(x, self.weight)

        # Graph convolution: Â (X W) → (*, N, D_out)
        if x.dim() == 3:
            # Batched: adj_norm is (N,N), support is (B, N, D_out)
            # Expand adj for batch matmul
            out = torch.matmul(adj_norm.unsqueeze(0), support)
        else:
            out = torch.matmul(adj_norm, support)

        if self.bias is not None:
            out = out + self.bias

        return out


class GCN(nn.Module):
    """
    Multi-layer Graph Convolutional Network.

    Architecture (for 3 layers with embed_dim=256):
        Layer 1: GCNLayer(256, 256) → ReLU → Dropout
        Layer 2: GCNLayer(256, 256) → ReLU → Dropout
        Layer 3: GCNLayer(256, 256) — no activation on final layer

    The output maintains the same dimensionality as the input embeddings,
    preserving spatial and structural information for the Graph Transformer.
    """

    def __init__(
        self,
        in_features: int = 256,
        hidden_features: int = 256,
        out_features: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        layers = []
        for i in range(num_layers):
            in_dim = in_features if i == 0 else hidden_features
            out_dim = out_features if i == num_layers - 1 else hidden_features
            layers.append(GCNLayer(in_dim, out_dim))

        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,           # (B, N, D) or (N, D)
        adj_norm: torch.Tensor,     # (N, N) normalised adjacency
    ) -> torch.Tensor:
        """
        Forward pass through all GCN layers.

        ReLU activation and dropout after each layer except the last.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, adj_norm)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x
