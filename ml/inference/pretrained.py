"""Inference adapter for the community pretrained image deepfake detector."""
from __future__ import annotations

from typing import Dict, Mapping, Protocol, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
MODEL_REVISION = "3a99ae26f52c7ac7c3a53103b6cf3a8b617f7093"
DETECTOR_NAME = "Community pretrained ViT (Deep-Fake-Detector-v2)"
MODEL_NOTICE = (
    "Experimental third-party image classifier; scores are uncalibrated and "
    "may not generalize to all media. Video results average sampled face frames."
)


class _ModelConfig(Protocol):
    id2label: Mapping[int | str, str]


class _ModelOutput(Protocol):
    logits: torch.Tensor


class _ImageClassifier(Protocol):
    config: _ModelConfig

    def to(self, device: str) -> "_ImageClassifier": ...

    def eval(self) -> "_ImageClassifier": ...

    def __call__(self, **inputs: torch.Tensor) -> _ModelOutput: ...


class _ImageProcessor(Protocol):
    def __call__(
        self, *, images: Image.Image, return_tensors: str
    ) -> Dict[str, torch.Tensor]: ...


def _resolve_class_indexes(id2label: Mapping[int | str, str]) -> Tuple[int, int]:
    labels = {int(index): label.casefold() for index, label in id2label.items()}
    fake_indexes = [
        index for index, label in labels.items()
        if "fake" in label or "deepfake" in label
    ]
    real_indexes = [
        index for index, label in labels.items()
        if "real" in label and index not in fake_indexes
    ]
    if len(fake_indexes) != 1 or len(real_indexes) != 1:
        raise ValueError(f"Unsupported pretrained model labels: {dict(id2label)}")
    return real_indexes[0], fake_indexes[0]


class PretrainedDeepfakeDetector:
    def __init__(
        self, model: _ImageClassifier, processor: _ImageProcessor, device: str
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.real_index, self.fake_index = _resolve_class_indexes(model.config.id2label)

    @classmethod
    def from_pretrained(cls, device: str) -> "PretrainedDeepfakeDetector":
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        model = AutoModelForImageClassification.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            use_safetensors=True,
            trust_remote_code=False,
        )
        processor = AutoImageProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=False,
        )
        model.to(device)
        model.eval()
        return cls(model, processor, device)

    @torch.inference_mode()
    def predict(self, face_bgr: np.ndarray) -> Dict[str, object]:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(
            images=Image.fromarray(face_rgb),
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        logits = self.model(**inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1)
        real_probability = float(probabilities[self.real_index].item())
        fake_probability = float(probabilities[self.fake_index].item())
        return {
            "prediction": int(fake_probability > real_probability),
            "real_probability": real_probability,
            "fake_probability": fake_probability,
            "raw_logits": logits.cpu().tolist(),
        }
