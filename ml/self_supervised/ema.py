"""
Exponential Moving Average (EMA) for Teacher Network Updates

The teacher network is updated as an exponential moving average of the
student network parameters — no gradient-based optimisation for the teacher.

    θ_teacher ← m * θ_teacher + (1 - m) * θ_student

where m is the momentum coefficient, typically starting at 0.996 and
annealed towards 1.0 using a cosine schedule over training.
"""
from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn


class EMATeacher:
    """
    Manage the teacher network as an EMA of the student.

    Usage:
        ema = EMATeacher(student_model, base_momentum=0.996, total_epochs=800)
        teacher = ema.teacher

        for epoch in range(800):
            for batch in dataloader:
                # train student ...
                ema.update(student_model, epoch)
    """

    def __init__(
        self,
        student: nn.Module,
        base_momentum: float = 0.996,
        final_momentum: float = 1.0,
        total_epochs: int = 800,
    ) -> None:
        self.base_momentum = base_momentum
        self.final_momentum = final_momentum
        self.total_epochs = total_epochs

        # Deep-copy student → teacher (detached, no grad)
        self.teacher = copy.deepcopy(student)
        for param in self.teacher.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, student: nn.Module, current_epoch: int) -> float:
        """
        Update teacher parameters via EMA.

        Momentum is cosine-annealed from base_momentum to final_momentum.

        Returns the current momentum value.
        """
        m = self._cosine_momentum(current_epoch)

        for teacher_param, student_param in zip(
            self.teacher.parameters(), student.parameters()
        ):
            teacher_param.data.mul_(m).add_(student_param.data, alpha=1.0 - m)

        # Also update buffers (e.g. BatchNorm running stats)
        for teacher_buf, student_buf in zip(
            self.teacher.buffers(), student.buffers()
        ):
            teacher_buf.data.copy_(student_buf.data)

        return m

    def _cosine_momentum(self, epoch: int) -> float:
        """
        Cosine annealing of momentum:
            m = final - (final - base) * (cos(π * epoch / total) + 1) / 2
        """
        if self.total_epochs <= 1:
            return self.base_momentum
        progress = epoch / (self.total_epochs - 1)
        progress = min(1.0, max(0.0, progress))
        return (
            self.final_momentum
            - (self.final_momentum - self.base_momentum)
            * (math.cos(math.pi * progress) + 1.0) / 2.0
        )
