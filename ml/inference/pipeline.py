"""
End-to-End Inference Pipeline

Orchestrates the complete deepfake detection workflow:
    Image/Video → Face Detection → Preprocessing → Model → Prediction + Explainability

This module is imported by the backend detection service.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from ml.face_detection.detector import FaceDetector
from ml.preprocessing.transforms import (
    image_to_tensor,
    load_image,
    prepare_face_for_model,
)
from ml.preprocessing.video_processor import VideoProcessor
from ml.inference.model import DeepfakeDetectionModel
from ml.inference.pretrained import (
    DETECTOR_NAME,
    MODEL_ID,
    MODEL_NOTICE,
    PretrainedDeepfakeDetector,
)

logger = logging.getLogger(__name__)


def _resolve_classifier_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Prefer the best trained checkpoint, with deterministic fallbacks."""
    for filename in ("classifier_best.pt", "classifier_final.pt"):
        checkpoint = checkpoint_dir / filename
        if checkpoint.is_file():
            return checkpoint

    candidates = sorted(
        checkpoint_dir.glob("classifier_*.pt"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    return candidates[0] if candidates else None


class ModelNotTrainedError(RuntimeError):
    """Raised when inference is requested without a trained classifier."""


class InferencePipeline:
    """
    Full inference pipeline for deepfake detection.

    Manages:
    - Device selection (CUDA / CPU)
    - Model loading from checkpoints
    - Face detection
    - Image preprocessing
    - Model inference
    - Heatmap generation (when explainability module is available)
    - Video frame processing and aggregation
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        pretrain_checkpoint_path: Optional[str] = None,
        image_size: int = 320,
        patch_size: int = 20,
        embed_dim: int = 256,
        vit_depth: int = 12,
        gcn_layers: int = 3,
        transformer_blocks: int = 3,
        knn_k: int = 8,
        confidence_threshold: float = 0.5,
        device: str = "auto",
        video_sample_rate: int = 1,
        video_max_frames: int = 50,
        checkpoint_dir: str = "./checkpoints",
    ) -> None:
        # Device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        logger.info("Inference pipeline using device: %s", self.device)

        self.image_size = image_size
        self.patch_size = patch_size
        self.confidence_threshold = confidence_threshold
        self.checkpoint_dir = Path(checkpoint_dir)
        self.pretrained_model_id = MODEL_ID
        self._pretrained_detector: Optional[PretrainedDeepfakeDetector] = None
        self._pretrained_detector_lock = threading.Lock()

        # Face detector
        self.face_detector = FaceDetector(device=self.device)

        # Video processor
        self.video_processor = VideoProcessor(
            sample_rate=video_sample_rate,
            max_frames=video_max_frames,
        )

        # Build model
        self.model = DeepfakeDetectionModel(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            vit_depth=vit_depth,
            gcn_layers=gcn_layers,
            transformer_blocks=transformer_blocks,
            knn_k=knn_k,
            freeze_vit=True,
        )

        # Load checkpoints if available
        self._model_loaded = False
        self._load_checkpoints(checkpoint_path, pretrain_checkpoint_path)

        self.model.to(self.device)
        self.model.eval()

    def _load_checkpoints(
        self,
        checkpoint_path: Optional[str],
        pretrain_checkpoint_path: Optional[str],
    ) -> None:
        """Load model checkpoints if available."""
        # Try loading full classifier checkpoint first
        if checkpoint_path:
            path = Path(checkpoint_path)
            if not path.is_file():
                raise FileNotFoundError(f"Classifier checkpoint not found: {path}")
            self._load_classifier_checkpoint(path)
            return

        # Try loading pretrained ViT
        if pretrain_checkpoint_path:
            self.model.load_pretrained_vit(pretrain_checkpoint_path)
            logger.info("Loaded pretrained ViT; classifier training is still required")
            return

        # Auto-discover checkpoints
        checkpoints_dir = self.checkpoint_dir
        if checkpoints_dir.exists():
            classifier_checkpoint = _resolve_classifier_checkpoint(checkpoints_dir)
            if classifier_checkpoint:
                self._load_classifier_checkpoint(classifier_checkpoint)

        if not self._model_loaded:
            logger.warning(
                "No trained classifier checkpoint found in %s",
                checkpoints_dir,
            )

    def _load_classifier_checkpoint(self, checkpoint_path: Path) -> None:
        """Load a complete classifier checkpoint or raise an explicit error."""
        state = torch.load(str(checkpoint_path), map_location="cpu")
        model_state = state.get("model_state_dict", state) if isinstance(state, dict) else state
        if not isinstance(model_state, dict):
            raise TypeError(f"Invalid classifier checkpoint format: {checkpoint_path}")
        self.model.load_state_dict(model_state)
        self._model_loaded = True
        logger.info("Loaded classifier checkpoint: %s", checkpoint_path)

    def _require_trained_model(self) -> None:
        if self._model_loaded:
            return
        if getattr(self, "pretrained_model_id", None):
            self._get_pretrained_detector()
            return
        raise ModelNotTrainedError(
            "No trained Graph Transformer classifier checkpoint is available. "
            "Ask an administrator to complete classifier training before running detection."
        )

    def _get_pretrained_detector(self) -> PretrainedDeepfakeDetector:
        if self._pretrained_detector is None:
            with self._pretrained_detector_lock:
                if self._pretrained_detector is None:
                    logger.info(
                        "Loading pretrained image detector %s at pinned revision",
                        self.pretrained_model_id,
                    )
                    self._pretrained_detector = (
                        PretrainedDeepfakeDetector.from_pretrained(self.device)
                    )
        return self._pretrained_detector

    @property
    def detector_name(self) -> str:
        if self._model_loaded:
            return "Project Graph Transformer"
        if getattr(self, "_pretrained_detector", None) is not None:
            return DETECTOR_NAME
        return "Project Graph Transformer"

    @property
    def detector_notice(self) -> Optional[str]:
        return (
            MODEL_NOTICE
            if getattr(self, "_pretrained_detector", None) is not None
            else None
        )

    @property
    def is_model_trained(self) -> bool:
        """Check whether a trained classifier is available for inference."""
        return (
            self._model_loaded
            or getattr(self, "_pretrained_detector", None) is not None
        )

    async def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Run deepfake detection on a single image.

        Returns dict with prediction results and metadata.
        """
        # Run synchronous inference in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._analyze_image_sync, image_path)

    def _analyze_image_sync(self, image_path: str) -> Dict[str, Any]:
        """Synchronous image analysis."""
        self._require_trained_model()
        start = time.time()

        # 1. Load image
        image = load_image(image_path)

        # 2. Face detection
        faces = self.face_detector.detect_and_crop(
            image, target_size=(self.image_size, self.image_size)
        )

        if not faces:
            raise ValueError(
                "No face detected in the image. "
                "Please upload a clear image containing a human face."
            )

        # Process each face (use first face for primary result)
        face_results = []
        face_crops_paths = []
        heatmap_path = None
        graph_data_path = None
        patch_relevance = []

        for i, face in enumerate(faces):
            result = self._run_inference(face.face_image)
            face_results.append(result)

            # Save face crop
            crop_path = self._save_face_crop(face.face_image, image_path, i)
            face_crops_paths.append(crop_path)

            # Generate heatmap for primary face
            if i == 0 and self._model_loaded:
                try:
                    heatmap_path = self._generate_heatmap(
                        face.face_image, result, image_path
                    )
                    graph_data_path = self._save_graph_data(result, image_path)
                    patch_relevance = result.get("patch_relevance", [])
                except Exception as e:
                    logger.warning("Heatmap generation failed: %s", e)

        # Use primary face result
        primary = face_results[0]
        elapsed_ms = int((time.time() - start) * 1000)

        return {
            "prediction": "FAKE" if primary["prediction"] == 1 else "REAL",
            "real_probability": float(primary["real_probability"]),
            "fake_probability": float(primary["fake_probability"]),
            "confidence": float(max(primary["real_probability"], primary["fake_probability"])),
            "raw_logits": primary["raw_logits"],
            "face_count": len(faces),
            "face_crops_paths": face_crops_paths,
            "heatmap_path": heatmap_path,
            "graph_data_path": graph_data_path,
            "patch_relevance": patch_relevance,
            "processing_time_ms": elapsed_ms,
            "model_trained": self.is_model_trained,
            "detector_name": self.detector_name,
            "detector_notice": self.detector_notice,
        }

    async def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """Run deepfake detection on a video (frame-level analysis + aggregation)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._analyze_video_sync, video_path)

    def _analyze_video_sync(self, video_path: str) -> Dict[str, Any]:
        """Synchronous video analysis."""
        self._require_trained_model()
        start = time.time()

        # 1. Validate video
        video_info = self.video_processor.validate_video(video_path)

        # 2. Extract frames
        frames = self.video_processor.extract_frames(video_path)

        if not frames:
            raise ValueError("Could not extract any frames from the video")

        # 3. Process each frame
        frame_results = []
        fake_probabilities = []
        total_faces = 0

        for frame_data in frames:
            try:
                faces = self.face_detector.detect_and_crop(
                    frame_data.image,
                    target_size=(self.image_size, self.image_size),
                )

                if faces:
                    # Use primary face
                    result = self._run_inference(faces[0].face_image)
                    fake_prob = float(result["fake_probability"])
                    fake_probabilities.append(fake_prob)
                    total_faces += len(faces)

                    frame_results.append({
                        "frame_index": frame_data.frame_index,
                        "timestamp_seconds": frame_data.timestamp_seconds,
                        "prediction": "FAKE" if result["prediction"] == 1 else "REAL",
                        "real_probability": float(result["real_probability"]),
                        "fake_probability": fake_prob,
                        "face_detected": True,
                    })
                else:
                    frame_results.append({
                        "frame_index": frame_data.frame_index,
                        "timestamp_seconds": frame_data.timestamp_seconds,
                        "prediction": "UNKNOWN",
                        "real_probability": 0.0,
                        "fake_probability": 0.0,
                        "face_detected": False,
                    })
            except Exception as e:
                logger.warning("Frame %d processing failed: %s", frame_data.frame_index, e)
                frame_results.append({
                    "frame_index": frame_data.frame_index,
                    "timestamp_seconds": frame_data.timestamp_seconds,
                    "prediction": "ERROR",
                    "real_probability": 0.0,
                    "fake_probability": 0.0,
                    "face_detected": False,
                })

        # 4. Aggregate: mean of frame-level fake probabilities
        if fake_probabilities:
            mean_fake_prob = sum(fake_probabilities) / len(fake_probabilities)
            mean_real_prob = 1.0 - mean_fake_prob
            agg_prediction = "FAKE" if mean_fake_prob >= self.confidence_threshold else "REAL"
        else:
            mean_fake_prob = 0.0
            mean_real_prob = 1.0
            agg_prediction = "UNKNOWN"

        # Identify suspicious timestamps
        suspicious = [
            fr["timestamp_seconds"]
            for fr in frame_results
            if fr["face_detected"] and fr["fake_probability"] >= self.confidence_threshold
        ]

        elapsed_ms = int((time.time() - start) * 1000)

        return {
            "aggregated": {
                "prediction": agg_prediction,
                "real_probability": mean_real_prob,
                "fake_probability": mean_fake_prob,
                "confidence": max(mean_real_prob, mean_fake_prob),
                "face_count": total_faces,
                "frames_processed": len(frames),
                "frames_with_faces": len(fake_probabilities),
                "aggregation_method": "mean_fake_probability",
                "detector_name": self.detector_name,
                "detector_notice": self.detector_notice,
            },
            "frames": frame_results,
            "suspicious_timestamps": suspicious,
            "video_info": video_info,
            "processing_time_ms": elapsed_ms,
            "model_trained": self.is_model_trained,
        }

    @torch.no_grad()
    def _run_inference(self, face_bgr: np.ndarray) -> Dict[str, Any]:
        """Run model inference on a single preprocessed face image."""
        if not self._model_loaded:
            return self._get_pretrained_detector().predict(face_bgr)

        # Preprocess
        tensor = prepare_face_for_model(face_bgr, self.image_size)
        batch = tensor.unsqueeze(0).to(self.device)

        # Forward pass
        output = self.model(batch)

        return {
            "prediction": int(output["prediction"][0].item()),
            "real_probability": float(output["real_probability"][0].item()),
            "fake_probability": float(output["fake_probability"][0].item()),
            "raw_logits": output["logits"][0].cpu().tolist(),
            "cls_embedding": output["cls_embedding"][0].cpu().numpy(),
            "node_embeddings": output["node_embeddings"][0].cpu().numpy(),
        }

    def _save_face_crop(
        self, face_bgr: np.ndarray, source_path: str, face_idx: int
    ) -> str:
        """Save a cropped face image to disk."""
        output_dir = Path("./uploads/faces")
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(source_path).stem
        filename = f"{stem}_face{face_idx}_{uuid.uuid4().hex[:8]}.png"
        path = output_dir / filename
        cv2.imwrite(str(path), face_bgr)
        return str(path)

    def _generate_heatmap(
        self,
        face_bgr: np.ndarray,
        inference_result: Dict[str, Any],
        source_path: str,
    ) -> Optional[str]:
        """Generate attention heatmap using explainability module."""
        try:
            from ml.explainability.relevancy import compute_gradient_weighted_relevancy
            from ml.explainability.heatmap import generate_heatmap_overlay

            face_tensor = prepare_face_for_model(
                face_bgr, self.image_size
            ).unsqueeze(0).to(self.device)
            relevancy = compute_gradient_weighted_relevancy(
                self.model,
                face_tensor,
                target_class=int(inference_result["prediction"]),
                grid_size=self.image_size // self.patch_size,
            )

            # Generate overlay
            output_dir = Path("./uploads/heatmaps")
            output_dir.mkdir(parents=True, exist_ok=True)

            stem = Path(source_path).stem
            filename = f"{stem}_heatmap_{uuid.uuid4().hex[:8]}.png"
            path = output_dir / filename

            generate_heatmap_overlay(
                face_bgr, relevancy, str(path),
                grid_size=self.image_size // self.patch_size,
            )
            return str(path)

        except ImportError:
            logger.debug("Explainability module not yet implemented")
            return None
        except Exception as e:
            logger.warning("Heatmap generation error: %s", e)
            return None

    def _save_graph_data(
        self, inference_result: Dict[str, Any], source_path: str
    ) -> Optional[str]:
        """Save graph structure data for visualization."""
        try:
            output_dir = Path("./uploads/graphs")
            output_dir.mkdir(parents=True, exist_ok=True)

            stem = Path(source_path).stem
            filename = f"{stem}_graph_{uuid.uuid4().hex[:8]}.json"
            path = output_dir / filename

            graph_info = self.model.graph_builder.get_adjacency_info()

            data = {
                "num_nodes": graph_info["num_nodes"],
                "num_edges": graph_info["num_edges"],
                "k": graph_info["k"],
                "avg_degree": graph_info["avg_degree"],
                "is_symmetric": graph_info["is_symmetric"],
            }

            with open(path, "w") as f:
                json.dump(data, f, indent=2)

            return str(path)
        except Exception as e:
            logger.warning("Graph data save failed: %s", e)
            return None


# Singleton (lazy-loaded)
_pipeline: Optional[InferencePipeline] = None
_pipeline_key: Optional[tuple[str, Optional[str], Optional[int], Optional[int]]] = None
_pipeline_lock = threading.Lock()


def get_pipeline(checkpoint_dir: str = "./checkpoints") -> InferencePipeline:
    """Return a cached pipeline, refreshing it when the newest checkpoint changes."""
    global _pipeline, _pipeline_key

    directory = Path(checkpoint_dir)
    checkpoint = _resolve_classifier_checkpoint(directory)
    key = (
        str(directory.resolve()),
        str(checkpoint.resolve()) if checkpoint else None,
        checkpoint.stat().st_mtime_ns if checkpoint else None,
        checkpoint.stat().st_size if checkpoint else None,
    )

    with _pipeline_lock:
        if _pipeline is None or _pipeline_key != key:
            _pipeline = InferencePipeline(
                checkpoint_path=str(checkpoint) if checkpoint else None,
                checkpoint_dir=str(directory),
            )
            _pipeline_key = key
        return _pipeline
