"""
Complete Deepfake Detection Model

Combines all components into a single end-to-end model:
    ViT Feature Extractor → Graph Construction → GCN → Graph Transformer → Classification

This module is used for:
1. Supervised classifier training (with frozen/unfrozen ViT)
2. Inference (full pipeline prediction)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from ml.feature_extractor.vit import VisionTransformer
from ml.graph.builder import GraphBuilder
from ml.gcn.model import GCN
from ml.transformer.classifier import GraphTransformerClassifier

logger = logging.getLogger(__name__)


class DeepfakeDetectionModel(nn.Module):
    """
    Full model: ViT → Graph → GCN → Graph Transformer → Real/Fake

    This is the complete architecture described in the research paper:
    "Self-Supervised Graph Transformer for Deepfake Detection"
    """

    def __init__(
        self,
        image_size: int = 320,
        patch_size: int = 20,
        embed_dim: int = 256,
        vit_depth: int = 12,
        vit_heads: int = 8,
        gcn_layers: int = 3,
        gcn_dropout: float = 0.1,
        transformer_blocks: int = 3,
        transformer_heads: int = 8,
        transformer_mlp_dim: int = 512,
        transformer_dropout: float = 0.1,
        knn_k: int = 8,
        num_classes: int = 2,
        freeze_vit: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.freeze_vit = freeze_vit

        grid_size = image_size // patch_size  # 16
        num_patches = grid_size * grid_size    # 256

        # 1. Vision Transformer feature extractor
        self.vit = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=vit_depth,
            num_heads=vit_heads,
        )

        # 2. Graph builder (non-trainable, precomputed topology)
        self.graph_builder = GraphBuilder(grid_size=grid_size, k=knn_k)

        # 3. Graph Convolutional Network
        self.gcn = GCN(
            in_features=embed_dim,
            hidden_features=embed_dim,
            out_features=embed_dim,
            num_layers=gcn_layers,
            dropout=gcn_dropout,
        )

        # 4. Graph Transformer Classifier
        self.classifier = GraphTransformerClassifier(
            num_nodes=num_patches,
            embed_dim=embed_dim,
            num_blocks=transformer_blocks,
            num_heads=transformer_heads,
            mlp_dim=transformer_mlp_dim,
            num_classes=num_classes,
            dropout=transformer_dropout,
        )

        if freeze_vit:
            self._freeze_vit()

    def _freeze_vit(self) -> None:
        """Freeze ViT parameters (use pretrained features without fine-tuning)."""
        for param in self.vit.parameters():
            param.requires_grad = False
        logger.info("ViT parameters frozen")

    def unfreeze_vit(self) -> None:
        """Unfreeze ViT for end-to-end fine-tuning."""
        for param in self.vit.parameters():
            param.requires_grad = True
        self.freeze_vit = False
        logger.info("ViT parameters unfrozen for fine-tuning")

    def forward(self, x: torch.Tensor) -> dict:
        """
        Full forward pass: Image → ViT → Graph → GCN → Transformer → Classification

        Args:
            x: (B, C, H, W) input image tensor (preprocessed face, 320×320)

        Returns:
            dict with all prediction outputs + intermediate representations
        """
        # 1. ViT feature extraction
        if self.freeze_vit:
            with torch.no_grad():
                vit_out = self.vit(x)
        else:
            vit_out = self.vit(x)

        patch_features = vit_out["patch_tokens"]  # (B, N, D)

        # 2. Graph construction (topology is fixed, features vary)
        graph = self.graph_builder.build_graph(patch_features)
        adj_norm = graph.adjacency_norm  # (N, N)

        # 3. GCN processing
        gcn_out = self.gcn(patch_features, adj_norm)  # (B, N, D)

        # 4. Graph Transformer classification
        result = self.classifier(gcn_out)

        # Add intermediate representations
        result["vit_patch_features"] = patch_features
        result["gcn_features"] = gcn_out
        result["graph_adjacency"] = graph.adjacency
        result["graph_adjacency_norm"] = adj_norm
        result["spatial_coords"] = graph.spatial_coords

        return result

    def load_pretrained_vit(self, checkpoint_path: str) -> None:
        """
        Load pretrained ViT weights from self-supervised pretraining.
        Only loads the ViT backbone, not the SSL projection heads.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"ViT checkpoint not found: {checkpoint_path}")

        state_dict = torch.load(str(path), map_location="cpu")

        # Handle checkpoint formats
        if "student" in state_dict:
            # From student-teacher checkpoint
            vit_state = {
                k.replace("student.", ""): v
                for k, v in state_dict["student"].items()
                if not k.startswith("student.cls_head") and not k.startswith("student.patch_head")
            }
        elif "vit" in state_dict:
            vit_state = state_dict["vit"]
        elif "model" in state_dict:
            vit_state = {
                k.replace("vit.", ""): v
                for k, v in state_dict["model"].items()
                if k.startswith("vit.")
            }
        else:
            vit_state = state_dict

        # Filter out projection heads (cls_head, patch_head)
        filtered = {
            k: v for k, v in vit_state.items()
            if "cls_head" not in k and "patch_head" not in k
        }

        missing, unexpected = self.vit.load_state_dict(filtered, strict=False)
        if missing:
            logger.warning("Missing keys when loading ViT: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys when loading ViT: %s", unexpected)

        logger.info("Loaded pretrained ViT from %s", checkpoint_path)

    def get_model_info(self) -> dict:
        """Return model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "architecture": "SelfSupervisedGraphTransformer",
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "vit_depth": self.vit.depth,
            "gcn_layers": self.gcn.num_layers,
            "transformer_blocks": self.classifier.num_blocks,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "vit_frozen": self.freeze_vit,
        }
