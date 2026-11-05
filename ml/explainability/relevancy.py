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
    if input_tensor.shape[0] != 1:
        raise ValueError("Gradient-weighted relevancy expects a single input image.")

    classifier = model.classifier
    classifier.set_attention_gradient_capture(True)
    model.zero_grad(set_to_none=True)
    try:
        output = model(input_tensor)
        if target_class is None:
            target_class = int(output["prediction"][0].item())
        if target_class < 0 or target_class >= output["logits"].shape[1]:
            raise ValueError(f"Invalid target class: {target_class}")

        output["logits"][0, target_class].backward()

        result = None
        for block in classifier.blocks:
            attn = block.attn_weights
            if attn is None or attn.grad is None:
                raise RuntimeError(
                    "Attention gradients were not captured for every Transformer block."
                )

            attn_avg = attn.detach()[0].mean(dim=0)
            grad_weight = attn.grad.detach()[0].mean(dim=0).abs()
            attn_weighted = attn_avg * grad_weight
            attn_with_id = attn_weighted + torch.eye(
                attn_weighted.size(0), device=attn_weighted.device
            )
            attn_norm = attn_with_id / attn_with_id.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)

            result = attn_norm if result is None else torch.matmul(result, attn_norm)

        if result is None:
            raise RuntimeError("The classifier did not produce attention maps.")

        cls_relevance = result[0, 1:]
        rel_min = cls_relevance.min()
        rel_max = cls_relevance.max()
        if rel_max > rel_min:
            cls_relevance = (cls_relevance - rel_min) / (rel_max - rel_min)
        else:
            cls_relevance = torch.zeros_like(cls_relevance)

        return cls_relevance.detach().cpu().numpy()
    finally:
        classifier.set_attention_gradient_capture(False)
        for block in classifier.blocks:
            block.attn_weights = None
        model.zero_grad(set_to_none=True)
