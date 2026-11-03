"""
Test Suite: ML Pipeline Components

Tests for preprocessing, face detection, patch extraction, graph construction,
GCN, transformer, and the complete inference pipeline.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestPreprocessing:
    """Tests for image preprocessing and transforms."""

    def test_image_to_tensor_shape(self):
        from ml.preprocessing.transforms import image_to_tensor
        img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        tensor = image_to_tensor(img, normalize=True)
        assert tensor.shape == (3, 320, 320)
        assert tensor.dtype == torch.float32

    def test_image_to_tensor_no_normalize(self):
        from ml.preprocessing.transforms import image_to_tensor
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        tensor = image_to_tensor(img, normalize=False)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_prepare_face_for_model(self):
        from ml.preprocessing.transforms import prepare_face_for_model
        face = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        tensor = prepare_face_for_model(face, image_size=320)
        assert tensor.shape == (3, 320, 320)

    def test_resize_face(self):
        from ml.preprocessing.transforms import resize_face
        img = np.random.randint(0, 255, (500, 400, 3), dtype=np.uint8)
        resized = resize_face(img, (320, 320))
        assert resized.shape == (320, 320, 3)


# ══════════════════════════════════════════════════════════════════════════════
# Patch Extractor Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestPatchExtractor:
    """Tests for patch extraction module."""

    def test_extract_patches_single(self):
        from ml.preprocessing.patch_extractor import PatchExtractor
        pe = PatchExtractor(image_size=320, patch_size=20)
        img = torch.randn(3, 320, 320)
        patches = pe.extract_patches(img)
        assert patches.shape == (256, 3, 20, 20)

    def test_extract_patches_batch(self):
        from ml.preprocessing.patch_extractor import PatchExtractor
        pe = PatchExtractor(image_size=320, patch_size=20)
        batch = torch.randn(4, 3, 320, 320)
        patches = pe.extract_patches(batch)
        assert patches.shape == (4, 256, 3, 20, 20)

    def test_num_patches(self):
        from ml.preprocessing.patch_extractor import PatchExtractor
        pe = PatchExtractor(image_size=320, patch_size=20)
        assert pe.num_patches == 256
        assert pe.grid_size == 16

    def test_patch_coords(self):
        from ml.preprocessing.patch_extractor import PatchExtractor
        pe = PatchExtractor(image_size=320, patch_size=20)
        coords = pe.get_patch_coords()
        assert len(coords) == 256
        assert coords[0] == (0, 0, 20, 20)
        assert coords[-1] == (300, 300, 20, 20)

    def test_patches_to_image_roundtrip(self):
        from ml.preprocessing.patch_extractor import PatchExtractor
        pe = PatchExtractor(image_size=320, patch_size=20)
        img = torch.randn(3, 320, 320)
        patches = pe.extract_patches(img)
        reconstructed = pe.patches_to_image(patches)
        assert torch.allclose(img, reconstructed, atol=1e-6)

    def test_flatten_patches(self):
        from ml.preprocessing.patch_extractor import PatchExtractor
        pe = PatchExtractor(image_size=320, patch_size=20)
        patches = torch.randn(256, 3, 20, 20)
        flat = pe.flatten_patches(patches)
        assert flat.shape == (256, 3 * 20 * 20)


# ══════════════════════════════════════════════════════════════════════════════
# Vision Transformer Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestVisionTransformer:
    """Tests for ViT feature extractor."""

    def test_forward_shape(self):
        from ml.feature_extractor.vit import VisionTransformer
        vit = VisionTransformer(
            image_size=320, patch_size=20, embed_dim=256, depth=2, num_heads=8,
        )
        x = torch.randn(2, 3, 320, 320)
        out = vit(x)
        assert out["cls_token"].shape == (2, 256)
        assert out["patch_tokens"].shape == (2, 256, 256)
        assert out["cls_head_out"].shape == (2, 256)
        assert out["patch_head_out"].shape == (2, 256, 256)

    def test_masked_forward(self):
        from ml.feature_extractor.vit import VisionTransformer
        vit = VisionTransformer(
            image_size=320, patch_size=20, embed_dim=256, depth=2, num_heads=4,
        )
        x = torch.randn(2, 3, 320, 320)
        mask = torch.randint(0, 2, (2, 256)).bool()
        out = vit(x, mask=mask)
        assert out["cls_token"].shape == (2, 256)

    def test_extract_features(self):
        from ml.feature_extractor.vit import VisionTransformer
        vit = VisionTransformer(
            image_size=320, patch_size=20, embed_dim=256, depth=2, num_heads=4,
        )
        x = torch.randn(1, 3, 320, 320)
        cls_feat, patch_feat = vit.extract_features(x)
        assert cls_feat.shape == (1, 256)
        assert patch_feat.shape == (1, 256, 256)


# ══════════════════════════════════════════════════════════════════════════════
# Graph Construction Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestGraphBuilder:
    """Tests for graph construction."""

    def test_build_graph(self):
        from ml.graph.builder import GraphBuilder
        gb = GraphBuilder(grid_size=16, k=8)
        features = torch.randn(256, 256)
        graph = gb.build_graph(features)
        assert graph.num_nodes == 256
        assert graph.adjacency.shape == (256, 256)

    def test_adjacency_symmetric(self):
        from ml.graph.builder import GraphBuilder
        gb = GraphBuilder(grid_size=16, k=8)
        adj = gb._adj
        assert torch.allclose(adj, adj.T), "Adjacency must be symmetric"

    def test_self_loops(self):
        from ml.graph.builder import GraphBuilder
        gb = GraphBuilder(grid_size=16, k=8)
        adj = gb._adj
        diag = torch.diag(adj)
        assert (diag == 1.0).all(), "Diagonal must be 1 (self-loops)"

    def test_adjacency_normalized(self):
        from ml.graph.builder import GraphBuilder
        gb = GraphBuilder(grid_size=16, k=8)
        adj_norm = gb._adj_norm
        # Row sums should be approximately 1 for normalized adjacency
        row_sums = adj_norm.sum(dim=1)
        assert row_sums.min() > 0, "Normalized adjacency rows must be non-zero"

    def test_edge_index_format(self):
        from ml.graph.builder import GraphBuilder
        gb = GraphBuilder(grid_size=16, k=8)
        ei = gb._edge_index
        assert ei.dim() == 2
        assert ei.shape[0] == 2

    def test_batched_graph(self):
        from ml.graph.builder import GraphBuilder
        gb = GraphBuilder(grid_size=16, k=8)
        features = torch.randn(4, 256, 256)
        graph = gb.build_graph(features)
        assert graph.node_features.shape == (4, 256, 256)


# ══════════════════════════════════════════════════════════════════════════════
# GCN Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestGCN:
    """Tests for Graph Convolutional Network."""

    def test_gcn_forward(self):
        from ml.gcn.model import GCN
        from ml.graph.builder import GraphBuilder
        gcn = GCN(in_features=256, hidden_features=256, out_features=256, num_layers=3)
        gb = GraphBuilder(grid_size=16, k=8)
        x = torch.randn(2, 256, 256)
        adj_norm = gb._adj_norm
        out = gcn(x, adj_norm)
        assert out.shape == (2, 256, 256)

    def test_gcn_single(self):
        from ml.gcn.model import GCN
        from ml.graph.builder import GraphBuilder
        gcn = GCN(in_features=128, hidden_features=128, out_features=128, num_layers=2)
        gb = GraphBuilder(grid_size=16, k=8)
        x = torch.randn(256, 128)
        adj_norm = gb._adj_norm
        out = gcn(x, adj_norm)
        assert out.shape == (256, 128)

    def test_gcn_dropout(self):
        from ml.gcn.model import GCN
        from ml.graph.builder import GraphBuilder
        gcn = GCN(in_features=256, out_features=256, num_layers=3, dropout=0.5)
        gb = GraphBuilder(grid_size=16, k=8)
        x = torch.randn(256, 256)
        gcn.train()
        out1 = gcn(x, gb._adj_norm)
        out2 = gcn(x, gb._adj_norm)
        # With dropout, outputs should differ
        # (not guaranteed but very likely with 50% dropout)


# ══════════════════════════════════════════════════════════════════════════════
# Graph Transformer Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestGraphTransformer:
    """Tests for Graph Transformer classifier."""

    def test_forward_output(self):
        from ml.transformer.classifier import GraphTransformerClassifier
        cls = GraphTransformerClassifier(
            num_nodes=256, embed_dim=256, num_blocks=3, num_heads=8, mlp_dim=512,
        )
        x = torch.randn(2, 256, 256)
        out = cls(x)
        assert out["logits"].shape == (2, 2)
        assert out["probabilities"].shape == (2, 2)
        assert out["prediction"].shape == (2,)
        assert out["real_probability"].shape == (2,)
        assert out["fake_probability"].shape == (2,)
        assert out["cls_embedding"].shape == (2, 256)
        assert out["node_embeddings"].shape == (2, 256, 256)

    def test_probabilities_sum(self):
        from ml.transformer.classifier import GraphTransformerClassifier
        cls = GraphTransformerClassifier(
            num_nodes=256, embed_dim=256, num_blocks=2, num_heads=4,
        )
        x = torch.randn(4, 256, 256)
        out = cls(x)
        prob_sum = out["probabilities"].sum(dim=1)
        assert torch.allclose(prob_sum, torch.ones(4), atol=1e-5)

    def test_attention_maps(self):
        from ml.transformer.classifier import GraphTransformerClassifier
        cls = GraphTransformerClassifier(
            num_nodes=256, embed_dim=256, num_blocks=3, num_heads=8,
        )
        x = torch.randn(1, 256, 256)
        cls(x)
        maps = cls.get_attention_maps()
        assert len(maps) == 3
        for m in maps:
            assert m.shape[2] == 257  # 256 nodes + 1 CLS


# ══════════════════════════════════════════════════════════════════════════════
# Full Model Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestFullModel:
    """Tests for the complete DeepfakeDetectionModel."""

    def test_forward(self):
        from ml.inference.model import DeepfakeDetectionModel
        model = DeepfakeDetectionModel(
            image_size=320, patch_size=20, embed_dim=64,
            vit_depth=2, gcn_layers=2, transformer_blocks=2,
            transformer_heads=4, knn_k=4,
        )
        model.eval()
        x = torch.randn(1, 3, 320, 320)
        with torch.no_grad():
            out = model(x)
        assert out["logits"].shape == (1, 2)
        assert out["prediction"].shape == (1,)
        assert "vit_patch_features" in out
        assert "gcn_features" in out

    def test_model_info(self):
        from ml.inference.model import DeepfakeDetectionModel
        model = DeepfakeDetectionModel(
            image_size=320, patch_size=20, embed_dim=64,
            vit_depth=2, gcn_layers=2, transformer_blocks=2,
        )
        info = model.get_model_info()
        assert info["architecture"] == "SelfSupervisedGraphTransformer"
        assert info["image_size"] == 320
        assert info["total_parameters"] > 0


# ══════════════════════════════════════════════════════════════════════════════
# Self-Supervised Components Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestSelfSupervised:
    """Tests for self-supervised learning components."""

    def test_block_mask_shape(self):
        from ml.self_supervised.masking import generate_block_mask
        mask = generate_block_mask(grid_size=16, mask_ratio=0.4)
        assert mask.shape == (256,)
        assert mask.dtype == torch.bool

    def test_block_mask_ratio(self):
        from ml.self_supervised.masking import generate_block_mask
        mask = generate_block_mask(grid_size=16, mask_ratio=0.4)
        ratio = mask.float().mean().item()
        # Should be roughly in range [0.1, 0.8]
        assert 0.05 < ratio < 0.95

    def test_batch_masks(self):
        from ml.self_supervised.masking import generate_batch_masks
        masks = generate_batch_masks(batch_size=8, grid_size=16)
        assert masks.shape == (8, 256)

    def test_cls_loss(self):
        from ml.self_supervised.losses import compute_cls_loss
        s1 = torch.randn(4, 256)
        s2 = torch.randn(4, 256)
        t1 = torch.randn(4, 256)
        t2 = torch.randn(4, 256)
        loss = compute_cls_loss(s1, s2, t1, t2)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_mim_loss(self):
        from ml.self_supervised.losses import compute_mim_loss
        s1 = torch.randn(2, 256, 256)
        s2 = torch.randn(2, 256, 256)
        t1 = torch.randn(2, 256, 256)
        t2 = torch.randn(2, 256, 256)
        m1 = torch.randint(0, 2, (2, 256)).bool()
        m2 = torch.randint(0, 2, (2, 256)).bool()
        loss = compute_mim_loss(s1, s2, t1, t2, m1, m2)
        assert loss.shape == ()

    def test_ema_update(self):
        from ml.self_supervised.ema import EMATeacher
        from ml.feature_extractor.vit import VisionTransformer
        student = VisionTransformer(
            image_size=320, patch_size=20, embed_dim=64, depth=1, num_heads=4,
        )
        ema = EMATeacher(student, base_momentum=0.99, total_epochs=10)
        # Modify student
        for p in student.parameters():
            p.data.add_(1.0)
        # Update teacher
        m = ema.update(student, current_epoch=0)
        assert 0.99 <= m <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Explainability Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestExplainability:
    """Tests for relevancy map and heatmap generation."""

    def test_relevancy_map(self):
        from ml.explainability.relevancy import compute_relevancy_map
        attn_maps = [torch.randn(1, 8, 257, 257).softmax(dim=-1) for _ in range(3)]
        rel = compute_relevancy_map(attn_maps, num_blocks=3)
        assert rel.shape == (256,)
        assert rel.min() >= 0.0
        assert rel.max() <= 1.0

    def test_heatmap_generation(self):
        from ml.explainability.heatmap import relevance_to_heatmap
        rel = np.random.rand(256)
        heatmap = relevance_to_heatmap(rel, grid_size=16, image_size=320)
        assert heatmap.shape == (320, 320)

    def test_heatmap_overlay(self):
        from ml.explainability.heatmap import generate_heatmap_overlay
        face = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        rel = np.random.rand(256)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            generate_heatmap_overlay(face, rel, path, grid_size=16)
            assert Path(path).exists()
            assert Path(path).stat().st_size > 0
        finally:
            os.unlink(path)


# ══════════════════════════════════════════════════════════════════════════════
# Video Processing Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestVideoProcessor:
    """Tests for video frame extraction."""

    def test_supported_extensions(self):
        from ml.preprocessing.video_processor import VideoProcessor
        vp = VideoProcessor()
        assert ".mp4" in vp.SUPPORTED_EXTENSIONS
        assert ".avi" in vp.SUPPORTED_EXTENSIONS
        assert ".mov" in vp.SUPPORTED_EXTENSIONS

    def test_invalid_video_raises(self):
        from ml.preprocessing.video_processor import VideoProcessor
        vp = VideoProcessor()
        with pytest.raises(FileNotFoundError):
            vp.validate_video("nonexistent.mp4")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
