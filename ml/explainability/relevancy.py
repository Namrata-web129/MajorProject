"""
Graph Transformer Relevancy Map

Implements gradient-weighted Transformer attention relevance
as described in the research paper.

The approach:
1. Extract attention maps from all Transformer blocks
2. Calculate gradients with respect to the predicted class
3. Combine attention values and gradient relevance
4. Average across attention heads
5. Add identity connections to avoid self-inhibition
6. Propagate relevance through Transformer layers
7. Map resulting relevance to graph nodes
8. Convert to spatial heatmap
"""
from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


def compute_relevancy_map(
    attention_maps: List[torch.Tensor],
    num_blocks: int = 3,
) -> np.ndarray:
    """
    Compute relevancy map from Transformer attention weights.

    Uses the rollout method with identity connections:
        R_i = (A_i + I) (averaged across heads, then normalised)
        R_total = R_1 × R_2 × ... × R_L

    Then extract the CLS row to get per-node relevance.

    Args:
        attention_maps: List of (B, H, N+1, N+1) attention tensors from each block
        num_blocks: Number of Transformer blocks

    Returns:
        relevance: (N,) numpy array of per-node relevance scores
    """
    if not attention_maps:
        logger.warning("No attention maps available for relevancy computation")
        return np.zeros(256)

    # Use first sample in batch
    result = None

    for attn in attention_maps:
        # attn: (B, H, L, L) where L = N+1 (nodes + CLS)
        # Average across heads
        attn_avg = attn[0].mean(dim=0)  # (L, L)

        # Add identity matrix (avoid self-inhibition)
        I = torch.eye(attn_avg.size(0), device=attn_avg.device)
        attn_with_id = attn_avg + I

        # Row-normalise
        attn_norm = attn_with_id / attn_with_id.sum(dim=-1, keepdim=True)

        if result is None:
            result = attn_norm
        else:
            result = torch.matmul(result, attn_norm)

    if result is None:
        return np.zeros(256)

    # Extract CLS row → relevance to each node
    # result: (L, L), row 0 is CLS
    cls_relevance = result[0, 1:]  # (N,) — skip CLS-to-CLS

    # Normalise to [0, 1]
    rel_min = cls_relevance.min()
    rel_max = cls_relevance.max()
    if rel_max > rel_min:
        cls_relevance = (cls_relevance - rel_min) / (rel_max - rel_min)
    else:
        cls_relevance = torch.zeros_like(cls_relevance)

    return cls_relevance.cpu().numpy()


def compute_gradient_weighted_relevancy(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: Optional[int] = None,
    grid_size: int = 16,
) -> np.ndarray:
    """
    Compute gradient-weighted attention relevancy map.

    This is the full approach from the research paper:
    1. Forward pass with gradient tracking
    2. Backward pass on target class logit
    3. Weight attention by gradients
    4. Propagate through layers

    Args:
        model: The full DeepfakeDetectionModel
        input_tensor: (1, C, H, W) single image tensor
        target_class: Target class for gradient (None = predicted class)
        grid_size: Patch grid size (16)

    Returns:
        relevance: (N,) numpy array of per-node relevance scores
    """
    model.eval()

    # Enable gradient computation
    input_tensor = input_tensor.requires_grad_(False)

    # Forward pass
    output = model(input_tensor)

    if target_class is None:
        target_class = output["prediction"][0].item()

    # Get logits for target class
    logit = output["logits"][0, target_class]

    # Backward pass
    model.zero_grad()
    logit.backward(retain_graph=True)

    # Get attention maps from the Graph Transformer classifier
    attn_maps = model.classifier.get_attention_maps()

    if not attn_maps:
        return np.zeros(grid_size * grid_size)

    # Gradient-weighted attention rollout
    result = None
    num_nodes = grid_size * grid_size + 1  # +1 for CLS

    for block_idx, block in enumerate(model.classifier.blocks):
        attn = block.attn_weights  # (B, H, L, L) — stored during forward

        if attn is None:
            continue

        # Get gradients for Q, K projections
        # We weight attention by gradient magnitudes
        attn_avg = attn[0].mean(dim=0)  # (L, L)

        # Gradient-weighted attention
        # Use gradient of attention w.r.t. class logit
        if attn.grad is not None:
            grad_weight = attn.grad[0].mean(dim=0).abs()
            attn_weighted = attn_avg * grad_weight
        else:
            attn_weighted = attn_avg

        # Add identity
        I = torch.eye(attn_weighted.size(0), device=attn_weighted.device)
        attn_with_id = attn_weighted + I

        # Row-normalise
        attn_norm = attn_with_id / attn_with_id.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        if result is None:
            result = attn_norm
        else:
            result = torch.matmul(result, attn_norm)

    if result is None:
        return np.zeros(grid_size * grid_size)

    # CLS row → node relevance
    cls_relevance = result[0, 1:].detach()

    # Normalise
    rel_min = cls_relevance.min()
    rel_max = cls_relevance.max()
    if rel_max > rel_min:
        cls_relevance = (cls_relevance - rel_min) / (rel_max - rel_min)

    return cls_relevance.cpu().numpy()
