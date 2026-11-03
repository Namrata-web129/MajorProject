"""
Student-Teacher Self-Distillation Architecture

Wraps the full self-supervised training setup:
- Student network (ViT) — receives masked views, trained by gradient descent
- Teacher network (ViT) — receives unmasked views, updated via EMA
- Multi-crop augmentations (two global views)
- Blockwise masking
- CLS + MIM loss computation
- Running center (DINO-style) to avoid collapse
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from ml.feature_extractor.vit import VisionTransformer
from ml.self_supervised.ema import EMATeacher
from ml.self_supervised.losses import compute_total_loss
from ml.self_supervised.masking import generate_batch_masks

logger = logging.getLogger(__name__)


class StudentTeacherFramework(nn.Module):
    """
    Self-Supervised Student-Teacher framework for ViT pretraining.

    Training procedure per iteration:
    1. Generate two augmented views of each image
    2. Generate blockwise masks for each view
    3. Pass masked views through student
    4. Pass unmasked views through teacher (no grad)
    5. Compute L_CLS (cross-view) + L_MIM (in-view masked)
    6. Backprop on student only
    7. EMA update on teacher
    """

    def __init__(
        self,
        image_size: int = 320,
        patch_size: int = 20,
        embed_dim: int = 256,
        vit_depth: int = 12,
        num_heads: int = 8,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        mask_ratio: float = 0.4,
        ema_momentum: float = 0.996,
        total_epochs: int = 800,
        center_momentum: float = 0.9,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.mask_ratio = mask_ratio
        self.center_momentum = center_momentum

        grid_size = image_size // patch_size  # 16

        # Student network
        self.student = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=vit_depth,
            num_heads=num_heads,
        )

        # Teacher network (EMA of student)
        self.ema = EMATeacher(
            self.student,
            base_momentum=ema_momentum,
            total_epochs=total_epochs,
        )
        self.teacher = self.ema.teacher

        # Running center vectors (DINO-style, prevents mode collapse)
        self.register_buffer(
            "cls_center", torch.zeros(1, embed_dim)
        )
        self.register_buffer(
            "patch_center", torch.zeros(1, embed_dim)
        )

        self.grid_size = grid_size

    def forward(
        self,
        view1: torch.Tensor,  # (B, C, H, W) augmented view 1
        view2: torch.Tensor,  # (B, C, H, W) augmented view 2
    ) -> dict:
        """
        Forward pass for self-supervised training.

        Returns dict with losses and diagnostics.
        """
        B = view1.shape[0]

        # Generate masks for each view
        mask1 = generate_batch_masks(
            B, self.grid_size, self.mask_ratio
        ).to(view1.device)
        mask2 = generate_batch_masks(
            B, self.grid_size, self.mask_ratio
        ).to(view2.device)

        # Student: receives MASKED views
        student_out1 = self.student(view1, mask=mask1)
        student_out2 = self.student(view2, mask=mask2)

        # Teacher: receives UNMASKED views (no gradient)
        with torch.no_grad():
            teacher_out1 = self.teacher(view1, mask=None)
            teacher_out2 = self.teacher(view2, mask=None)

        # Compute losses
        loss_dict = compute_total_loss(
            student_out_view1=student_out1,
            student_out_view2=student_out2,
            teacher_out_view1=teacher_out1,
            teacher_out_view2=teacher_out2,
            mask_view1=mask1,
            mask_view2=mask2,
            student_temp=self.student_temp,
            teacher_temp=self.teacher_temp,
            cls_center=self.cls_center,
            patch_center=self.patch_center,
        )

        # Update running centers (DINO-style)
        with torch.no_grad():
            batch_cls_center = torch.cat([
                teacher_out1["cls_head_out"],
                teacher_out2["cls_head_out"],
            ], dim=0).mean(dim=0, keepdim=True)

            batch_patch_center = torch.cat([
                teacher_out1["patch_head_out"].mean(dim=1),
                teacher_out2["patch_head_out"].mean(dim=1),
            ], dim=0).mean(dim=0, keepdim=True)

            m = self.center_momentum
            self.cls_center = m * self.cls_center + (1 - m) * batch_cls_center
            self.patch_center = m * self.patch_center + (1 - m) * batch_patch_center

        return loss_dict

    def update_teacher(self, epoch: int) -> float:
        """Update teacher network via EMA. Returns current momentum."""
        return self.ema.update(self.student, epoch)

    def get_feature_extractor(self) -> VisionTransformer:
        """Return the trained student ViT for downstream use."""
        return self.student
