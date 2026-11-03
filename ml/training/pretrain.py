"""
Self-Supervised ViT Pretrainer

Implements the student-teacher self-distillation pretraining loop:
1. Load unlabeled images
2. Generate two augmented views
3. Apply blockwise masking
4. Student processes masked views; teacher processes unmasked views
5. Compute L_CLS + L_MIM losses
6. Update student via gradient descent
7. Update teacher via EMA

Learning rate: lr = 5e-4 * batch_size / 256
Default: 800 epochs (configurable)
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ml.self_supervised.student_teacher import StudentTeacherFramework
from ml.training.dataset_loader import create_ssl_dataloader

logger = logging.getLogger(__name__)


class SelfSupervisedPretrainer:
    """
    Self-supervised pretraining manager.

    Creates the student-teacher framework, dataset loader,
    optimizer, and runs the training loop with checkpoint saving.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self.config = config
        self.stop_event = stop_event or threading.Event()

        # Extract config
        self.dataset_path = config["dataset_path"]
        self.batch_size = config.get("batch_size", 32)
        self.epochs = config.get("epochs", 100)
        self.seed = config.get("seed", 42)
        self.save_every = config.get("save_every", 10)
        self.mask_ratio = config.get("mask_ratio", 0.4)
        self.teacher_momentum = config.get("teacher_momentum", 0.996)

        # Learning rate: lr = 5e-4 * batch_size / 256
        self.lr = config.get("learning_rate") or (5e-4 * self.batch_size / 256)

        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Checkpoints directory
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Set seed
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    async def train(
        self,
        epoch_callback: Optional[Callable[..., Coroutine]] = None,
    ) -> str:
        """
        Run self-supervised pretraining.

        Args:
            epoch_callback: async function(epoch, metrics) called after each epoch

        Returns:
            Path to final checkpoint
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._train_sync, epoch_callback
        )

    def _train_sync(
        self,
        epoch_callback: Optional[Callable] = None,
    ) -> str:
        """Synchronous training loop."""
        logger.info("Starting self-supervised pretraining on %s", self.device)
        logger.info("Config: batch=%d, epochs=%d, lr=%.6f", self.batch_size, self.epochs, self.lr)

        # Create dataloader
        dataloader = create_ssl_dataloader(
            dataset_path=self.dataset_path,
            batch_size=self.batch_size,
            image_size=320,
            num_workers=min(4, os.cpu_count() or 1),
        )

        # Create framework
        framework = StudentTeacherFramework(
            image_size=320,
            patch_size=20,
            embed_dim=256,
            vit_depth=12,
            student_temp=0.1,
            teacher_temp=0.04,
            mask_ratio=self.mask_ratio,
            ema_momentum=self.teacher_momentum,
            total_epochs=self.epochs,
            device=self.device,
        ).to(self.device)

        # Optimizer (student only)
        optimizer = AdamW(
            framework.student.parameters(),
            lr=self.lr,
            weight_decay=0.04,
            betas=(0.9, 0.999),
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)

        # Check for resume checkpoint
        start_epoch = 0
        resume_path = self.checkpoint_dir / "pretrain_latest.pt"
        if resume_path.exists():
            try:
                ckpt = torch.load(str(resume_path), map_location=self.device)
                framework.student.load_state_dict(ckpt["student"])
                framework.teacher.load_state_dict(ckpt["teacher"])
                optimizer.load_state_dict(ckpt["optimizer"])
                start_epoch = ckpt.get("epoch", 0) + 1
                logger.info("Resumed from epoch %d", start_epoch)
            except Exception as e:
                logger.warning("Could not resume: %s", e)

        checkpoint_path = ""

        for epoch in range(start_epoch, self.epochs):
            if self.stop_event.is_set():
                logger.info("Stop signal received at epoch %d", epoch)
                break

            epoch_loss = 0.0
            epoch_cls_loss = 0.0
            epoch_mim_loss = 0.0
            num_batches = 0

            framework.train()

            for batch_idx, (view1, view2) in enumerate(dataloader):
                if self.stop_event.is_set():
                    break

                view1 = view1.to(self.device)
                view2 = view2.to(self.device)

                # Forward
                loss_dict = framework(view1, view2)
                loss = loss_dict["total_loss"]

                # Backward
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(framework.student.parameters(), max_norm=3.0)

                optimizer.step()

                # EMA teacher update
                framework.update_teacher(epoch)

                epoch_loss += loss.item()
                epoch_cls_loss += loss_dict["cls_loss"].item()
                epoch_mim_loss += loss_dict["mim_loss"].item()
                num_batches += 1

            scheduler.step()

            # Metrics
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                avg_cls = epoch_cls_loss / num_batches
                avg_mim = epoch_mim_loss / num_batches
            else:
                avg_loss = avg_cls = avg_mim = 0.0

            current_lr = scheduler.get_last_lr()[0]

            logger.info(
                "Epoch %d/%d — loss=%.4f (cls=%.4f, mim=%.4f) lr=%.6f",
                epoch + 1, self.epochs, avg_loss, avg_cls, avg_mim, current_lr,
            )

            # Callback
            if epoch_callback:
                metrics = {
                    "train_loss": avg_loss,
                    "train_cls_loss": avg_cls,
                    "train_mim_loss": avg_mim,
                    "learning_rate": current_lr,
                }
                try:
                    coro = epoch_callback(epoch + 1, metrics)
                    if asyncio.iscoroutine(coro):
                        loop = asyncio.new_event_loop()
                        loop.run_until_complete(coro)
                        loop.close()
                except Exception as e:
                    logger.warning("Epoch callback failed: %s", e)

            # Save checkpoint
            if (epoch + 1) % self.save_every == 0 or epoch == self.epochs - 1:
                checkpoint_path = str(
                    self.checkpoint_dir / f"pretrain_epoch{epoch + 1}.pt"
                )
                torch.save({
                    "epoch": epoch,
                    "student": framework.student.state_dict(),
                    "teacher": framework.teacher.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": self.config,
                }, checkpoint_path)

                # Also save as latest
                torch.save({
                    "epoch": epoch,
                    "student": framework.student.state_dict(),
                    "teacher": framework.teacher.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": self.config,
                }, str(resume_path))

                logger.info("Checkpoint saved: %s", checkpoint_path)

        # Save final ViT extractor separately
        final_path = str(self.checkpoint_dir / "pretrain_vit_final.pt")
        torch.save(
            framework.student.state_dict(),
            final_path,
        )
        logger.info("Final ViT extractor saved: %s", final_path)

        return checkpoint_path or final_path
