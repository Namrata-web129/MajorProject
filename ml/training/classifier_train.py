"""
Supervised Graph Transformer Classifier Training

Trains the full detection pipeline (ViT → Graph → GCN → Transformer)
on labeled real/fake data.

Supports:
- Loading pretrained ViT backbone
- Frozen or unfrozen backbone fine-tuning
- Training with validation
- Best model checkpoint saving
- Reproducible with configurable seed
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

from ml.inference.model import DeepfakeDetectionModel
from ml.training.dataset_loader import create_dataloaders

logger = logging.getLogger(__name__)


class GraphTransformerTrainer:
    """
    Supervised classifier trainer.

    Pipeline: image → ViT features → Graph → GCN → Graph Transformer → REAL/FAKE
    """

    def __init__(
        self,
        config: Dict[str, Any],
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self.config = config
        self.stop_event = stop_event or threading.Event()

        self.dataset_path = config["dataset_path"]
        self.batch_size = config.get("batch_size", 32)
        self.epochs = config.get("epochs", 50)
        self.lr = config.get("learning_rate", 1e-4)
        self.seed = config.get("seed", 42)
        self.save_every = config.get("save_every", 5)
        self.freeze_backbone = config.get("freeze_backbone", False)
        self.pretrain_checkpoint = config.get("pretrain_checkpoint")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.checkpoint_dir = Path(config.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    async def train(
        self,
        epoch_callback: Optional[Callable[..., Coroutine]] = None,
    ) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._train_sync, epoch_callback
        )

    def _train_sync(
        self,
        epoch_callback: Optional[Callable] = None,
    ) -> str:
        logger.info("Starting classifier training on %s", self.device)

        # Create dataloaders
        loaders = create_dataloaders(
            dataset_path=self.dataset_path,
            batch_size=self.batch_size,
            image_size=320,
            num_workers=min(4, os.cpu_count() or 1),
        )

        if "train" not in loaders:
            raise ValueError(f"No training data found at {self.dataset_path}/train/")

        train_loader = loaders["train"]
        val_loader = loaders.get("val")

        # Build model
        model = DeepfakeDetectionModel(
            image_size=320,
            patch_size=20,
            embed_dim=256,
            vit_depth=12,
            gcn_layers=3,
            transformer_blocks=3,
            knn_k=8,
            freeze_vit=self.freeze_backbone,
        )

        # Load pretrained ViT if available
        if self.pretrain_checkpoint and Path(self.pretrain_checkpoint).exists():
            model.load_pretrained_vit(self.pretrain_checkpoint)
            logger.info("Loaded pretrained ViT from %s", self.pretrain_checkpoint)

        model.to(self.device)

        # Loss & optimizer
        criterion = nn.CrossEntropyLoss()

        # Only train parameters that require grad
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=self.lr, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-7)

        best_val_loss = float("inf")
        best_checkpoint_path = ""

        for epoch in range(self.epochs):
            if self.stop_event.is_set():
                logger.info("Stop signal at epoch %d", epoch)
                break

            # ── Train ─────────────────────────────────────────────
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                if self.stop_event.is_set():
                    break

                images = images.to(self.device)
                labels = labels.to(self.device)

                output = model(images)
                loss = criterion(output["logits"], labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                preds = output["prediction"]
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            scheduler.step()

            num_batches = max(1, batch_idx + 1) if train_total > 0 else 1
            avg_train_loss = train_loss / num_batches
            train_acc = train_correct / max(train_total, 1)

            # ── Validate ──────────────────────────────────────────
            val_loss_avg = None
            val_acc = None

            if val_loader:
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(self.device)
                        labels = labels.to(self.device)

                        output = model(images)
                        loss = criterion(output["logits"], labels)

                        val_loss += loss.item()
                        preds = output["prediction"]
                        val_correct += (preds == labels).sum().item()
                        val_total += labels.size(0)

                val_batches = max(1, len(val_loader))
                val_loss_avg = val_loss / val_batches
                val_acc = val_correct / max(val_total, 1)

            current_lr = scheduler.get_last_lr()[0]

            logger.info(
                "Epoch %d/%d — train_loss=%.4f train_acc=%.3f "
                "val_loss=%s val_acc=%s lr=%.6f",
                epoch + 1, self.epochs,
                avg_train_loss, train_acc,
                f"{val_loss_avg:.4f}" if val_loss_avg is not None else "N/A",
                f"{val_acc:.3f}" if val_acc is not None else "N/A",
                current_lr,
            )

            # Callback
            if epoch_callback:
                metrics = {
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss_avg,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "learning_rate": current_lr,
                }
                try:
                    coro = epoch_callback(epoch + 1, metrics)
                    if asyncio.iscoroutine(coro):
                        loop = asyncio.new_event_loop()
                        loop.run_until_complete(coro)
                        loop.close()
                except Exception as e:
                    logger.warning("Epoch callback error: %s", e)

            # Save best model
            check_val = val_loss_avg if val_loss_avg is not None else avg_train_loss
            if check_val < best_val_loss:
                best_val_loss = check_val
                best_checkpoint_path = str(
                    self.checkpoint_dir / "classifier_best.pt"
                )
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": val_loss_avg,
                    "train_loss": avg_train_loss,
                    "config": self.config,
                }, best_checkpoint_path)
                logger.info("Best model saved at epoch %d (loss=%.4f)", epoch + 1, check_val)

            # Periodic checkpoints
            if (epoch + 1) % self.save_every == 0:
                periodic_path = str(
                    self.checkpoint_dir / f"classifier_epoch{epoch + 1}.pt"
                )
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": self.config,
                }, periodic_path)

        # Save final model
        final_path = str(self.checkpoint_dir / "classifier_final.pt")
        torch.save({
            "epoch": self.epochs,
            "model_state_dict": model.state_dict(),
            "config": self.config,
        }, final_path)

        return best_checkpoint_path or final_path
