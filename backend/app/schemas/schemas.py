"""
DeepFake Detection System – Backend
Pydantic Schemas for Detection, Analysis, Model, Training, Evaluation, Explainability
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Analysis / Detection ───────────────────────────────────────────────────────
class AnalysisSummary(BaseModel):
    id: str
    status: str
    error_message: Optional[str] = None
    prediction: Optional[str] = None
    real_probability: Optional[float] = None
    fake_probability: Optional[float] = None
    confidence: Optional[float] = None
    face_count: Optional[int] = None
    file_name: Optional[str] = None
    file_type: Optional[str] = None
    created_at: datetime
    processing_duration_ms: Optional[int] = None

    model_config = {"from_attributes": True}


class VideoFrameResultSchema(BaseModel):
    frame_index: int
    timestamp_seconds: float
    prediction: str
    real_probability: float
    fake_probability: float
    face_detected: bool

    model_config = {"from_attributes": True}


class DetectionResultSchema(BaseModel):
    prediction: str
    real_probability: float
    fake_probability: float
    confidence: float
    raw_logits: Optional[List[float]] = None
    detector_name: Optional[str] = None
    detector_notice: Optional[str] = None
    face_count: int
    heatmap_available: bool = False
    graph_data_available: bool = False
    patch_relevance: Optional[List[float]] = None

    model_config = {"from_attributes": True}


class FullAnalysisResponse(BaseModel):
    analysis: AnalysisSummary
    detection: Optional[DetectionResultSchema] = None
    video_frames: Optional[List[VideoFrameResultSchema]] = None
    suspicious_timestamps: Optional[List[float]] = None
    model_version: Optional[str] = None

    model_config = {"from_attributes": True}


class PaginatedHistory(BaseModel):
    items: List[AnalysisSummary]
    total: int
    page: int
    page_size: int
    has_next: bool


# ── Model Info ────────────────────────────────────────────────────────────────
class ModelInfoResponse(BaseModel):
    id: str
    name: str
    architecture: str
    embed_dim: int
    patch_size: int
    image_size: int
    gcn_layers: int
    transformer_blocks: int
    attention_heads: int
    knn_k: int
    training_status: str
    checkpoint_path: Optional[str]
    pretrain_checkpoint_path: Optional[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ModelStatusResponse(BaseModel):
    model_loaded: bool
    pretrained_fallback_available: bool
    training_status: str
    device: str
    checkpoint_exists: bool
    pretrain_checkpoint_exists: bool
    active_model_id: Optional[str]


# ── Training ──────────────────────────────────────────────────────────────────
class PretrainConfig(BaseModel):
    dataset_path: str
    batch_size: int = Field(default=32, ge=1, le=512)
    epochs: int = Field(default=100, ge=1, le=800)
    learning_rate: Optional[float] = None   # auto-scaled if None
    teacher_momentum: float = Field(default=0.996, ge=0.9, le=0.9999)
    mask_ratio: float = Field(default=0.75, ge=0.1, le=0.95)
    save_every: int = Field(default=10, ge=1)
    seed: int = 42
    model_name: str = "SelfSupervisedViT_v1"


class ClassifierTrainConfig(BaseModel):
    dataset_path: str
    pretrain_checkpoint: Optional[str] = None
    batch_size: int = Field(default=32, ge=1, le=512)
    epochs: int = Field(default=50, ge=1, le=500)
    learning_rate: float = Field(default=1e-4, gt=0)
    freeze_backbone: bool = False
    save_every: int = Field(default=5, ge=1)
    seed: int = 42
    model_name: str = "GraphTransformer_v1"


class TrainingJobResponse(BaseModel):
    job_id: str
    status: str
    message: str


class TrainingStatusResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    current_epoch: int
    total_epochs: int
    progress_pct: float
    best_val_loss: Optional[float]
    started_at: Optional[datetime]
    metrics: List[Dict[str, Any]]

    model_config = {"from_attributes": True}


# ── Evaluation ────────────────────────────────────────────────────────────────
class EvaluationRequest(BaseModel):
    model_version_id: str
    eval_type: str = "in_dataset"   # in_dataset | cross_dataset | robustness | ablation
    dataset_path: str
    dataset_name: str


class EvaluationResultSchema(BaseModel):
    id: str
    eval_type: str
    dataset_name: str
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    auc_roc: Optional[float]
    false_positive_rate: Optional[float]
    false_negative_rate: Optional[float]
    evaluated_at: datetime

    model_config = {"from_attributes": True}


# ── Explainability ────────────────────────────────────────────────────────────
class ExplainabilityResponse(BaseModel):
    analysis_id: str
    prediction: str
    confidence: float
    heatmap_base64: Optional[str] = None
    patch_relevance: Optional[List[float]] = None
    graph_nodes: Optional[int] = None
    graph_edges: Optional[int] = None
    attention_weights: Optional[List[List[float]]] = None
    explanation_text: str = ""

    model_config = {"from_attributes": True}
