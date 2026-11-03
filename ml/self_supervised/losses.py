"""
Self-Supervised Loss Functions

Implements the two loss functions from the research paper:

1. L_CLS — Cross-view CLS Self-Distillation:
   Cross-entropy between teacher CLS distribution (from view A)
   and student CLS distribution (from view B), and vice versa.

2. L_MIM — In-view Masked Patch Self-Distillation:
   Cross-entropy between teacher and student patch distributions,
   summed only over masked positions.

Total objective: L = L_CLS + L_MIM

Uses temperature-scaled probability distributions and numerically
stable cross-entropy calculations.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def temperature_softmax(
    logits: torch.Tensor,
    temperature: float,
    center: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute temperature-scaled softmax distribution.

    Args:
        logits: (B, D) or (B, N, D) raw outputs
        temperature: Temperature scalar (higher = softer distribution)
        center: Optional centering vector (used for teacher, like in DINO)

    Returns:
        Probability distribution of same shape as logits
    """
    if center is not None:
        logits = logits - center
    return F.softmax(logits / temperature, dim=-1)


def compute_cls_loss(
    student_cls_view1: torch.Tensor,  # (B, D) student CLS from view 1
    student_cls_view2: torch.Tensor,  # (B, D) student CLS from view 2
    teacher_cls_view1: torch.Tensor,  # (B, D) teacher CLS from view 1
    teacher_cls_view2: torch.Tensor,  # (B, D) teacher CLS from view 2
    student_temp: float = 0.1,
    teacher_temp: float = 0.04,
    teacher_center: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    L_CLS: Cross-view self-distillation between CLS tokens.

    Computes:
        loss = - sum( p_teacher(view1) * log(p_student(view2)) )
             + - sum( p_teacher(view2) * log(p_student(view1)) )

    This is averaged over the batch.
    """
    # Teacher distributions (no gradient)
    with torch.no_grad():
        t_prob1 = temperature_softmax(teacher_cls_view1, teacher_temp, teacher_center)
        t_prob2 = temperature_softmax(teacher_cls_view2, teacher_temp, teacher_center)

    # Student distributions
    s_log_prob1 = F.log_softmax(student_cls_view1 / student_temp, dim=-1)
    s_log_prob2 = F.log_softmax(student_cls_view2 / student_temp, dim=-1)

    # Cross-view cross-entropy
    # Teacher view 1 → Student view 2
    loss_12 = -torch.sum(t_prob1 * s_log_prob2, dim=-1).mean()
    # Teacher view 2 → Student view 1
    loss_21 = -torch.sum(t_prob2 * s_log_prob1, dim=-1).mean()

    return (loss_12 + loss_21) / 2.0


def compute_mim_loss(
    student_patch_view1: torch.Tensor,  # (B, N, D)
    student_patch_view2: torch.Tensor,  # (B, N, D)
    teacher_patch_view1: torch.Tensor,  # (B, N, D)
    teacher_patch_view2: torch.Tensor,  # (B, N, D)
    mask_view1: torch.Tensor,           # (B, N) bool — True = masked
    mask_view2: torch.Tensor,           # (B, N) bool — True = masked
    student_temp: float = 0.1,
    teacher_temp: float = 0.04,
    teacher_center: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    L_MIM: In-view masked patch self-distillation.

    For each view, compute cross-entropy between teacher and student
    patch distributions only at MASKED positions.

    This is an in-view objective: teacher and student operate on the
    same view, but the student sees the masked version.
    """
    loss = torch.tensor(0.0, device=student_patch_view1.device)
    count = 0

    for s_patches, t_patches, mask in [
        (student_patch_view1, teacher_patch_view1, mask_view1),
        (student_patch_view2, teacher_patch_view2, mask_view2),
    ]:
        # Only use masked positions
        # mask: (B, N) bool
        if mask.sum() == 0:
            continue

        B, N, D = s_patches.shape

        # Teacher distributions (no gradient) — at masked positions
        with torch.no_grad():
            t_logits = t_patches  # (B, N, D)
            if teacher_center is not None:
                t_logits = t_logits - teacher_center
            t_probs = F.softmax(t_logits / teacher_temp, dim=-1)

        # Student log-probabilities
        s_log_probs = F.log_softmax(s_patches / student_temp, dim=-1)

        # Compute per-token cross-entropy: (B, N)
        per_token_loss = -torch.sum(t_probs * s_log_probs, dim=-1)

        # Mask: only keep masked positions
        mask_float = mask.float()  # (B, N)
        masked_loss = (per_token_loss * mask_float).sum()

        n_masked = mask_float.sum().clamp(min=1.0)
        loss = loss + masked_loss / n_masked
        count += 1

    if count > 0:
        loss = loss / count

    return loss


def compute_total_loss(
    student_out_view1: dict,
    student_out_view2: dict,
    teacher_out_view1: dict,
    teacher_out_view2: dict,
    mask_view1: torch.Tensor,
    mask_view2: torch.Tensor,
    student_temp: float = 0.1,
    teacher_temp: float = 0.04,
    cls_center: torch.Tensor | None = None,
    patch_center: torch.Tensor | None = None,
) -> dict:
    """
    Compute total self-supervised objective: L = L_CLS + L_MIM

    Args:
        student_out_view1/2: ViT forward outputs for each view (masked student)
        teacher_out_view1/2: ViT forward outputs for each view (unmasked teacher)
        mask_view1/2: (B, N) boolean masks for each view
        student_temp: Student softmax temperature
        teacher_temp: Teacher softmax temperature (should be lower → sharper)
        cls_center: Running center for CLS tokens (DINO-style)
        patch_center: Running center for patch tokens

    Returns:
        dict with 'total_loss', 'cls_loss', 'mim_loss'
    """
    cls_loss = compute_cls_loss(
        student_cls_view1=student_out_view1["cls_head_out"],
        student_cls_view2=student_out_view2["cls_head_out"],
        teacher_cls_view1=teacher_out_view1["cls_head_out"],
        teacher_cls_view2=teacher_out_view2["cls_head_out"],
        student_temp=student_temp,
        teacher_temp=teacher_temp,
        teacher_center=cls_center,
    )

    mim_loss = compute_mim_loss(
        student_patch_view1=student_out_view1["patch_head_out"],
        student_patch_view2=student_out_view2["patch_head_out"],
        teacher_patch_view1=teacher_out_view1["patch_head_out"],
        teacher_patch_view2=teacher_out_view2["patch_head_out"],
        mask_view1=mask_view1,
        mask_view2=mask_view2,
        student_temp=student_temp,
        teacher_temp=teacher_temp,
        teacher_center=patch_center,
    )

    total_loss = cls_loss + mim_loss

    return {
        "total_loss": total_loss,
        "cls_loss": cls_loss,
        "mim_loss": mim_loss,
    }
