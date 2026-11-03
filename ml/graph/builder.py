"""
Deepfake Graph Construction Module

Converts ViT patch embeddings into an undirected graph G = (V, E).
- V: 256 graph nodes (one per patch)
- E: Connections between spatially nearest neighbours (K=8 KNN)
- Symmetric adjacency matrix with self-loops and normalization:
    Ã = A + I
    Â = D̃^(-1/2) Ã D̃^(-1/2)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DeepfakeGraph:
    """A graph constructed from facial patch embeddings."""
    node_features: torch.Tensor      # (N, D) — patch feature embeddings
    adjacency: torch.Tensor           # (N, N) — binary adjacency matrix
    adjacency_norm: torch.Tensor      # (N, N) — normalised adjacency Â
    edge_index: torch.Tensor          # (2, E) — COO edge index format
    spatial_coords: torch.Tensor      # (N, 2) — (row, col) grid coordinates
    patch_indices: torch.Tensor       # (N,) — patch indices
    num_nodes: int = 256
    k: int = 8


class GraphBuilder:
    """
    Build a KNN-based spatial graph from patch embeddings.

    For a 16×16 grid of patches, each patch is connected to its K=8
    spatially nearest neighbours based on grid distance.
    """

    def __init__(
        self,
        grid_size: int = 16,
        k: int = 8,
    ) -> None:
        self.grid_size = grid_size
        self.num_nodes = grid_size * grid_size  # 256
        self.k = k

        # Precompute spatial coordinates (row, col) for each patch
        coords = []
        for row in range(grid_size):
            for col in range(grid_size):
                coords.append((row, col))
        self.coords = torch.tensor(coords, dtype=torch.float32)  # (N, 2)

        # Precompute the adjacency matrix (topology is fixed for all images)
        self._adj, self._edge_index = self._build_knn_adjacency()
        self._adj_norm = self._normalize_adjacency(self._adj)

    def build_graph(self, patch_features: torch.Tensor) -> DeepfakeGraph:
        """
        Construct a DeepfakeGraph from patch feature embeddings.

        Args:
            patch_features: (N, D) or (B, N, D) patch embeddings from ViT

        Returns:
            DeepfakeGraph (for single image) or list of graphs (for batch)
        """
        if patch_features.dim() == 3:
            # Batch mode — build one graph per sample
            # For efficiency, topology is shared; only features differ
            B, N, D = patch_features.shape
            assert N == self.num_nodes, (
                f"Expected {self.num_nodes} patches, got {N}"
            )
            # Return list for batch processing
            # In practice, the same topology applies so we share adjacency
            return DeepfakeGraph(
                node_features=patch_features,  # (B, N, D)
                adjacency=self._adj.to(patch_features.device),
                adjacency_norm=self._adj_norm.to(patch_features.device),
                edge_index=self._edge_index.to(patch_features.device),
                spatial_coords=self.coords.to(patch_features.device),
                patch_indices=torch.arange(N, device=patch_features.device),
                num_nodes=N,
                k=self.k,
            )

        # Single image
        N, D = patch_features.shape
        assert N == self.num_nodes, f"Expected {self.num_nodes} patches, got {N}"

        return DeepfakeGraph(
            node_features=patch_features,
            adjacency=self._adj.to(patch_features.device),
            adjacency_norm=self._adj_norm.to(patch_features.device),
            edge_index=self._edge_index.to(patch_features.device),
            spatial_coords=self.coords.to(patch_features.device),
            patch_indices=torch.arange(N, device=patch_features.device),
            num_nodes=N,
            k=self.k,
        )

    def _build_knn_adjacency(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build a K-nearest-neighbour adjacency matrix based on spatial distance.

        Returns:
            adjacency: (N, N) symmetric binary adjacency matrix (with self-loops)
            edge_index: (2, E) COO format edge indices
        """
        N = self.num_nodes
        coords = self.coords  # (N, 2)

        # Pairwise Euclidean distance between all pairs
        dist = torch.cdist(coords, coords, p=2)  # (N, N)

        # For each node, find K nearest neighbours (excluding self)
        # Set self-distance to inf temporarily
        dist_no_self = dist.clone()
        dist_no_self.fill_diagonal_(float('inf'))

        # Top-K nearest per node
        _, knn_indices = torch.topk(dist_no_self, k=self.k, dim=1, largest=False)
        # knn_indices: (N, K)

        # Build symmetric adjacency matrix
        adj = torch.zeros(N, N, dtype=torch.float32)
        for i in range(N):
            for j in knn_indices[i]:
                adj[i, j.item()] = 1.0
                adj[j.item(), i] = 1.0  # Symmetric

        # Add self-loops: Ã = A + I
        adj_tilde = adj + torch.eye(N)
        # Clip to binary
        adj_tilde = (adj_tilde > 0).float()

        # Build COO edge index
        edge_src, edge_dst = torch.nonzero(adj_tilde, as_tuple=True)
        edge_index = torch.stack([edge_src, edge_dst], dim=0)  # (2, E)

        return adj_tilde, edge_index

    def _normalize_adjacency(self, adj_tilde: torch.Tensor) -> torch.Tensor:
        """
        Symmetric adjacency normalization:
            Â = D̃^(-1/2) Ã D̃^(-1/2)

        where Ã = A + I (adjacency with self-loops)
        and D̃ is the degree matrix of Ã.
        """
        # Degree matrix
        degree = adj_tilde.sum(dim=1)  # (N,)
        # D^(-1/2)
        d_inv_sqrt = torch.zeros_like(degree)
        mask = degree > 0
        d_inv_sqrt[mask] = 1.0 / torch.sqrt(degree[mask])
        # D^(-1/2) as diagonal matrix
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        # Â = D^(-1/2) Ã D^(-1/2)
        adj_norm = D_inv_sqrt @ adj_tilde @ D_inv_sqrt
        return adj_norm

    def get_adjacency_info(self) -> dict:
        """Return graph statistics for debugging/display."""
        N = self.num_nodes
        E = self._edge_index.shape[1]
        density = E / (N * N)
        degrees = self._adj.sum(dim=1)
        return {
            "num_nodes": N,
            "num_edges": E,
            "k": self.k,
            "density": density,
            "avg_degree": degrees.mean().item(),
            "min_degree": degrees.min().item(),
            "max_degree": degrees.max().item(),
            "is_symmetric": torch.allclose(self._adj, self._adj.T),
        }
