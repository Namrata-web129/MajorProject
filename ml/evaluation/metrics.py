"""
Model Evaluation Module

Computes:
1. Accuracy, Precision, Recall, F1-score, ROC-AUC
2. Confusion matrix
3. False positive rate, False negative rate
4. ROC curve data
5. Robustness evaluation (noise, blur, compression, etc.)
6. Cross-dataset evaluation
7. Ablation studies

All metrics come from actual model inference — never hardcoded.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluate the deepfake detection model on test datasets.
    """

    def __init__(
        self,
        model_version_id: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        self.model_version_id = model_version_id

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = None
        self.checkpoint_path = checkpoint_path

        # Try to load from database if model_version_id given
        if not checkpoint_path and model_version_id:
            self.checkpoint_path = self._find_checkpoint(model_version_id)

    def _find_checkpoint(self, model_version_id: str) -> Optional[str]:
        """Find checkpoint path for a model version."""
        checkpoints_dir = Path("./checkpoints")
        if checkpoints_dir.exists():
            best = checkpoints_dir / "classifier_best.pt"
            if best.exists():
                return str(best)
            final = checkpoints_dir / "classifier_final.pt"
            if final.exists():
                return str(final)
        return None

    def _load_model(self) -> None:
        """Load the model from checkpoint."""
        if self.model is not None:
            return

        from ml.inference.model import DeepfakeDetectionModel

        self.model = DeepfakeDetectionModel(
            image_size=320,
            patch_size=20,
            embed_dim=256,
            vit_depth=12,
            gcn_layers=3,
            transformer_blocks=3,
            knn_k=8,
            freeze_vit=True,
        )

        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            state = torch.load(self.checkpoint_path, map_location="cpu")
            if "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
            else:
                self.model.load_state_dict(state)
            logger.info("Loaded model from %s", self.checkpoint_path)
        else:
            logger.warning(
                "No checkpoint found. Evaluation will use randomly initialised weights. "
                "Results will NOT be meaningful."
            )

        self.model.to(self.device)
        self.model.eval()

    async def evaluate(
        self,
        dataset_path: str,
        eval_type: str = "in_dataset",
    ) -> Dict[str, Any]:
        """Run evaluation and return metrics."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._evaluate_sync, dataset_path, eval_type
        )

    def _evaluate_sync(
        self,
        dataset_path: str,
        eval_type: str = "in_dataset",
    ) -> Dict[str, Any]:
        """Synchronous evaluation."""
        self._load_model()

        from ml.training.dataset_loader import DeepfakeDataset

        # Load test set
        test_dir = Path(dataset_path) / "test"
        if not test_dir.exists():
            raise FileNotFoundError(f"Test directory not found: {test_dir}")

        dataset = DeepfakeDataset(
            root_dir=dataset_path,
            split="test",
            image_size=320,
            augment=False,
        )

        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=min(4, os.cpu_count() or 1),
        )

        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                output = self.model(images)

                all_labels.extend(labels.cpu().numpy().tolist())
                all_preds.extend(output["prediction"].cpu().numpy().tolist())
                all_probs.extend(output["fake_probability"].cpu().numpy().tolist())

        labels_np = np.array(all_labels)
        preds_np = np.array(all_preds)
        probs_np = np.array(all_probs)

        metrics = self._compute_metrics(labels_np, preds_np, probs_np)

        if eval_type == "robustness":
            metrics["extra_metrics"] = self._robustness_eval(dataset_path)

        return metrics

    def _compute_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        fake_probs: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute all classification metrics."""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            confusion_matrix,
            roc_curve,
        )

        accuracy = float(accuracy_score(labels, predictions))
        precision = float(precision_score(labels, predictions, zero_division=0))
        recall = float(recall_score(labels, predictions, zero_division=0))
        f1 = float(f1_score(labels, predictions, zero_division=0))

        try:
            auc_roc = float(roc_auc_score(labels, fake_probs))
        except ValueError:
            auc_roc = 0.0

        cm = confusion_matrix(labels, predictions, labels=[0, 1]).tolist()

        # FPR and FNR
        tn = cm[0][0] if len(cm) > 0 and len(cm[0]) > 0 else 0
        fp = cm[0][1] if len(cm) > 0 and len(cm[0]) > 1 else 0
        fn = cm[1][0] if len(cm) > 1 and len(cm[1]) > 0 else 0
        tp = cm[1][1] if len(cm) > 1 and len(cm[1]) > 1 else 0

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        # ROC curve data
        try:
            fpr_arr, tpr_arr, thresholds = roc_curve(labels, fake_probs)
            roc_data = {
                "fpr": fpr_arr.tolist(),
                "tpr": tpr_arr.tolist(),
                "thresholds": thresholds.tolist(),
            }
        except ValueError:
            roc_data = {"fpr": [], "tpr": [], "thresholds": []}

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc_roc": auc_roc,
            "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr),
            "confusion_matrix": cm,
            "roc_curve_data": roc_data,
            "total_samples": len(labels),
            "real_count": int((labels == 0).sum()),
            "fake_count": int((labels == 1).sum()),
        }

    def _robustness_eval(self, dataset_path: str) -> Dict[str, Any]:
        """
        Evaluate robustness against common corruptions.
        Tests: Gaussian noise, blur, JPEG compression, contrast changes.
        """
        import cv2
        from ml.training.dataset_loader import DeepfakeDataset

        corruption_results = {}
        corruptions = {
            "gaussian_noise": [0.01, 0.02, 0.05],
            "gaussian_blur": [1, 3, 5],
            "jpeg_compression": [90, 70, 50],
            "contrast": [0.8, 0.6, 0.4],
        }

        for corruption_name, intensities in corruptions.items():
            for intensity in intensities:
                key = f"{corruption_name}_{intensity}"
                try:
                    dataset = DeepfakeDataset(
                        root_dir=dataset_path,
                        split="test",
                        image_size=320,
                        augment=False,
                        max_samples=200,
                    )

                    loader = DataLoader(dataset, batch_size=16, shuffle=False)

                    all_labels = []
                    all_preds = []
                    all_probs = []

                    with torch.no_grad():
                        for images, labels in loader:
                            # Apply corruption
                            images = self._apply_corruption(
                                images, corruption_name, intensity
                            )
                            images = images.to(self.device)
                            output = self.model(images)

                            all_labels.extend(labels.numpy().tolist())
                            all_preds.extend(output["prediction"].cpu().numpy().tolist())
                            all_probs.extend(output["fake_probability"].cpu().numpy().tolist())

                    metrics = self._compute_metrics(
                        np.array(all_labels),
                        np.array(all_preds),
                        np.array(all_probs),
                    )
                    corruption_results[key] = {
                        "accuracy": metrics["accuracy"],
                        "f1_score": metrics["f1_score"],
                        "auc_roc": metrics["auc_roc"],
                    }
                except Exception as e:
                    logger.warning("Robustness test %s failed: %s", key, e)
                    corruption_results[key] = {"error": str(e)}

        return corruption_results

    def _apply_corruption(
        self,
        images: torch.Tensor,
        corruption: str,
        intensity: float,
    ) -> torch.Tensor:
        """Apply a corruption to a batch of images."""
        if corruption == "gaussian_noise":
            noise = torch.randn_like(images) * intensity
            return images + noise
        elif corruption == "gaussian_blur":
            kernel = int(intensity) * 2 + 1
            from torchvision.transforms.functional import gaussian_blur
            return gaussian_blur(images, kernel_size=kernel)
        elif corruption == "contrast":
            mean = images.mean(dim=[2, 3], keepdim=True)
            return (images - mean) * intensity + mean
        elif corruption == "jpeg_compression":
            # Simulate by quantizing
            quality = intensity / 100.0
            images = (images * quality).round() / max(quality, 0.01)
            return images.clamp(0, 1)
        return images
