from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ml.inference.pretrained import PretrainedDeepfakeDetector, _resolve_class_indexes


class _FakeProcessor:
    def __call__(self, *, images, return_tensors):
        assert images.mode == "RGB"
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros(1, 3, 224, 224)}


class _FakeModel:
    class config:
        id2label = {"0": "Realism", "1": "Deepfake"}

    def __call__(self, **inputs):
        assert "pixel_values" in inputs
        return type("Output", (), {"logits": torch.tensor([[0.0, 3.0]])})()


def test_pretrained_model_label_mapping():
    assert _resolve_class_indexes({"0": "Realism", "1": "Deepfake"}) == (0, 1)

    with pytest.raises(ValueError, match="Unsupported pretrained model labels"):
        _resolve_class_indexes({"0": "class_a", "1": "class_b"})


def test_pretrained_model_predicts_from_rgb_face_crop():
    detector = PretrainedDeepfakeDetector(
        _FakeModel(),
        _FakeProcessor(),
        device="cpu",
    )

    result = detector.predict(np.zeros((40, 40, 3), dtype=np.uint8))

    assert result["prediction"] == 1
    assert result["fake_probability"] > result["real_probability"]
    assert len(result["raw_logits"]) == 2


def test_pipeline_uses_pretrained_detector_when_graph_checkpoint_is_missing(monkeypatch):
    from ml.inference.pipeline import InferencePipeline

    pipeline = object.__new__(InferencePipeline)
    pipeline._model_loaded = False
    pipeline.pretrained_model_id = "published-model"
    pipeline._pretrained_detector = None
    pipeline._pretrained_detector_lock = threading.Lock()

    class FakeDetector:
        @classmethod
        def from_pretrained(cls, device):
            return cls()

        def predict(self, face_bgr):
            return {"prediction": 0}

    pipeline.device = "cpu"
    monkeypatch.setattr(
        PretrainedDeepfakeDetector,
        "from_pretrained",
        FakeDetector.from_pretrained,
    )
    pipeline._require_trained_model()
    assert pipeline._run_inference(np.zeros((8, 8, 3), dtype=np.uint8)) == {
        "prediction": 0
    }


def test_image_analysis_includes_pretrained_model_provenance(monkeypatch):
    import ml.inference.pipeline as inference_pipeline

    pipeline = object.__new__(inference_pipeline.InferencePipeline)
    pipeline._model_loaded = False
    pipeline.pretrained_model_id = "published-model"
    pipeline._pretrained_detector = PretrainedDeepfakeDetector(
        _FakeModel(),
        _FakeProcessor(),
        device="cpu",
    )
    pipeline.image_size = 40
    pipeline.face_detector = SimpleNamespace(
        detect_and_crop=lambda image, target_size: [
            SimpleNamespace(face_image=np.zeros((40, 40, 3), dtype=np.uint8))
        ]
    )
    monkeypatch.setattr(
        inference_pipeline,
        "load_image",
        lambda path: np.zeros((40, 40, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        pipeline,
        "_save_face_crop",
        lambda face, source, index: "face.png",
    )

    result = pipeline._analyze_image_sync("sample.png")

    assert result["prediction"] == "FAKE"
    assert result["detector_name"] == "Community pretrained ViT (Deep-Fake-Detector-v2)"
    assert result["detector_notice"]
    assert result["heatmap_path"] is None


def test_video_analysis_includes_pretrained_model_provenance():
    from ml.inference.pipeline import InferencePipeline

    frame = SimpleNamespace(
        image=np.zeros((40, 40, 3), dtype=np.uint8),
        frame_index=3,
        timestamp_seconds=1.5,
    )
    pipeline = object.__new__(InferencePipeline)
    pipeline._model_loaded = False
    pipeline.pretrained_model_id = "published-model"
    pipeline._pretrained_detector = PretrainedDeepfakeDetector(
        _FakeModel(),
        _FakeProcessor(),
        device="cpu",
    )
    pipeline.image_size = 40
    pipeline.confidence_threshold = 0.5
    pipeline.face_detector = SimpleNamespace(
        detect_and_crop=lambda image, target_size: [
            SimpleNamespace(face_image=np.zeros((40, 40, 3), dtype=np.uint8))
        ]
    )
    pipeline.video_processor = SimpleNamespace(
        validate_video=lambda path: {"duration": 3.0},
        extract_frames=lambda path: [frame],
    )

    result = pipeline._analyze_video_sync("sample.mp4")

    assert result["aggregated"]["prediction"] == "FAKE"
    assert result["aggregated"]["detector_name"] == (
        "Community pretrained ViT (Deep-Fake-Detector-v2)"
    )
    assert result["aggregated"]["detector_notice"]
    assert result["frames"][0]["timestamp_seconds"] == 1.5
