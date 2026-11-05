from __future__ import annotations

import pytest
import torch


class TestInferenceCheckpoint:
    def test_prefers_best_classifier_checkpoint(self, tmp_path):
        from ml.inference.pipeline import _resolve_classifier_checkpoint

        latest = tmp_path / "classifier_epoch50.pt"
        best = tmp_path / "classifier_best.pt"
        latest.touch()
        best.touch()

        assert _resolve_classifier_checkpoint(tmp_path) == best

    def test_inference_rejects_untrained_classifier(self):
        from ml.inference.pipeline import InferencePipeline, ModelNotTrainedError

        pipeline = object.__new__(InferencePipeline)
        pipeline._model_loaded = False

        with pytest.raises(ModelNotTrainedError, match="classifier checkpoint"):
            pipeline._analyze_image_sync("unused.jpg")

        with pytest.raises(ModelNotTrainedError, match="classifier checkpoint"):
            pipeline._analyze_video_sync("unused.mp4")

    def test_loads_complete_classifier_checkpoint(self, tmp_path):
        from ml.inference.model import DeepfakeDetectionModel
        from ml.inference.pipeline import InferencePipeline

        config = {
            "image_size": 40,
            "patch_size": 20,
            "embed_dim": 32,
            "vit_depth": 1,
            "vit_heads": 4,
            "gcn_layers": 1,
            "transformer_blocks": 1,
            "transformer_heads": 4,
            "knn_k": 1,
        }
        source = DeepfakeDetectionModel(**config)
        target = DeepfakeDetectionModel(**config)
        with torch.no_grad():
            source.classifier.cls_token.fill_(0.75)

        checkpoint = tmp_path / "classifier_test.pt"
        torch.save({"model_state_dict": source.state_dict()}, checkpoint)

        pipeline = object.__new__(InferencePipeline)
        pipeline.model = target
        pipeline._model_loaded = False
        pipeline._load_classifier_checkpoint(checkpoint)

        assert pipeline.is_model_trained
        assert torch.equal(target.classifier.cls_token, source.classifier.cls_token)

    def test_explicit_missing_checkpoint_fails(self, tmp_path):
        from ml.inference.pipeline import InferencePipeline

        pipeline = object.__new__(InferencePipeline)
        pipeline._model_loaded = False

        with pytest.raises(FileNotFoundError, match="Classifier checkpoint not found"):
            pipeline._load_checkpoints(str(tmp_path / "missing.pt"), None)


def test_gradient_weighted_relevancy_uses_captured_attention_gradients():
    from ml.explainability.relevancy import compute_gradient_weighted_relevancy
    from ml.inference.model import DeepfakeDetectionModel

    model = DeepfakeDetectionModel(
        image_size=40,
        patch_size=20,
        embed_dim=32,
        vit_depth=1,
        vit_heads=4,
        gcn_layers=1,
        transformer_blocks=2,
        transformer_heads=4,
        knn_k=1,
    )
    model.eval()

    relevance = compute_gradient_weighted_relevancy(
        model, torch.randn(1, 3, 40, 40), grid_size=2
    )

    assert relevance.shape == (4,)
    assert torch.isfinite(torch.from_numpy(relevance)).all()
    assert (relevance >= 0).all()
    assert relevance.max() > relevance.min()
    assert not any(block.capture_attention_gradients for block in model.classifier.blocks)
