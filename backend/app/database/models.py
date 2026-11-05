"""
DeepFake Detection System – Backend
All SQLAlchemy ORM Models (10 entities)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return str(uuid.uuid4())


# ─── 1. Users ─────────────────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    role: Mapped[str] = mapped_column(String(20), default="user", nullable=False)  # "user" | "admin"
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationships
    uploaded_files: Mapped[list["UploadedFile"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    analysis_history: Mapped[list["AnalysisHistory"]] = relationship(back_populates="user", cascade="all, delete-orphan")


# ─── 2. UploadedFiles ─────────────────────────────────────────────────────────
class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_filename: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[str] = mapped_column(String(20), nullable=False)   # "image" | "video"
    mime_type: Mapped[str] = mapped_column(String(100), nullable=True)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)       # bytes
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="uploaded_files")
    analysis: Mapped[Optional["AnalysisHistory"]] = relationship(back_populates="file", uselist=False)


# ─── 3. AnalysisHistory ───────────────────────────────────────────────────────
class AnalysisHistory(Base):
    __tablename__ = "analysis_history"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    file_id: Mapped[str] = mapped_column(String(36), ForeignKey("uploaded_files.id", ondelete="CASCADE"), nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="pending")   # pending | processing | completed | failed
    processing_duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    model_version_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("model_versions.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="analysis_history")
    file: Mapped["UploadedFile"] = relationship(back_populates="analysis")
    detection_result: Mapped[Optional["DetectionResult"]] = relationship(back_populates="analysis", uselist=False, cascade="all, delete-orphan")
    video_frame_results: Mapped[list["VideoFrameResult"]] = relationship(back_populates="analysis", cascade="all, delete-orphan")
    model_version: Mapped[Optional["ModelVersion"]] = relationship()


# ─── 4. DetectionResults ──────────────────────────────────────────────────────
class DetectionResult(Base):
    __tablename__ = "detection_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    analysis_id: Mapped[str] = mapped_column(String(36), ForeignKey("analysis_history.id", ondelete="CASCADE"), nullable=False, unique=True)
    prediction: Mapped[str] = mapped_column(String(10), nullable=False)   # "REAL" | "FAKE"
    real_probability: Mapped[float] = mapped_column(Float, nullable=False)
    fake_probability: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    raw_logits: Mapped[Optional[str]] = mapped_column(Text, nullable=True)   # JSON string
    detector_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    detector_notice: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    face_count: Mapped[int] = mapped_column(Integer, default=1)
    face_crops_paths: Mapped[Optional[str]] = mapped_column(Text, nullable=True)   # JSON list of paths
    heatmap_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    graph_data_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    patch_relevance: Mapped[Optional[str]] = mapped_column(Text, nullable=True)   # JSON
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # Relationships
    analysis: Mapped["AnalysisHistory"] = relationship(back_populates="detection_result")


# ─── 5. VideoFrameResults ─────────────────────────────────────────────────────
class VideoFrameResult(Base):
    __tablename__ = "video_frame_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    analysis_id: Mapped[str] = mapped_column(String(36), ForeignKey("analysis_history.id", ondelete="CASCADE"), nullable=False)
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    prediction: Mapped[str] = mapped_column(String(10), nullable=False)
    real_probability: Mapped[float] = mapped_column(Float, nullable=False)
    fake_probability: Mapped[float] = mapped_column(Float, nullable=False)
    face_detected: Mapped[bool] = mapped_column(Boolean, default=True)
    frame_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    # Relationships
    analysis: Mapped["AnalysisHistory"] = relationship(back_populates="video_frame_results")


# ─── 6. ModelVersions ─────────────────────────────────────────────────────────
class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    architecture: Mapped[str] = mapped_column(String(100), default="SelfSupervisedGraphTransformer")
    embed_dim: Mapped[int] = mapped_column(Integer, default=256)
    patch_size: Mapped[int] = mapped_column(Integer, default=20)
    image_size: Mapped[int] = mapped_column(Integer, default=320)
    gcn_layers: Mapped[int] = mapped_column(Integer, default=3)
    transformer_blocks: Mapped[int] = mapped_column(Integer, default=3)
    attention_heads: Mapped[int] = mapped_column(Integer, default=8)
    knn_k: Mapped[int] = mapped_column(Integer, default=8)
    checkpoint_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    pretrain_checkpoint_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    training_status: Mapped[str] = mapped_column(String(30), default="untrained")  # untrained | pretraining | training | trained | failed
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationships
    training_jobs: Mapped[list["TrainingJob"]] = relationship(back_populates="model_version", cascade="all, delete-orphan")
    evaluation_results: Mapped[list["EvaluationResult"]] = relationship(back_populates="model_version", cascade="all, delete-orphan")


# ─── 7. TrainingJobs ──────────────────────────────────────────────────────────
class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    model_version_id: Mapped[str] = mapped_column(String(36), ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False)
    job_type: Mapped[str] = mapped_column(String(30), nullable=False)  # "pretrain" | "classifier"
    status: Mapped[str] = mapped_column(String(30), default="queued")  # queued | running | completed | failed | stopped
    config: Mapped[Optional[str]] = mapped_column(Text, nullable=True)   # JSON config
    dataset_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    dataset_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    current_epoch: Mapped[int] = mapped_column(Integer, default=0)
    total_epochs: Mapped[int] = mapped_column(Integer, default=0)
    best_val_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # Relationships
    model_version: Mapped["ModelVersion"] = relationship(back_populates="training_jobs")
    metrics: Mapped[list["TrainingMetric"]] = relationship(back_populates="job", cascade="all, delete-orphan")


# ─── 8. TrainingMetrics ───────────────────────────────────────────────────────
class TrainingMetric(Base):
    __tablename__ = "training_metrics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    job_id: Mapped[str] = mapped_column(String(36), ForeignKey("training_jobs.id", ondelete="CASCADE"), nullable=False)
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    train_loss: Mapped[float] = mapped_column(Float, nullable=False)
    val_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    train_cls_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    train_mim_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    learning_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # Relationships
    job: Mapped["TrainingJob"] = relationship(back_populates="metrics")


# ─── 9. EvaluationResults ─────────────────────────────────────────────────────
class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    model_version_id: Mapped[str] = mapped_column(String(36), ForeignKey("model_versions.id", ondelete="CASCADE"), nullable=False)
    eval_type: Mapped[str] = mapped_column(String(50), nullable=False)   # "in_dataset" | "cross_dataset" | "robustness" | "ablation"
    dataset_name: Mapped[str] = mapped_column(String(100), nullable=False)
    accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precision: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recall: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    f1_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    auc_roc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    false_positive_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    false_negative_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confusion_matrix: Mapped[Optional[str]] = mapped_column(Text, nullable=True)   # JSON
    roc_curve_data: Mapped[Optional[str]] = mapped_column(Text, nullable=True)    # JSON
    extra_metrics: Mapped[Optional[str]] = mapped_column(Text, nullable=True)     # JSON
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # Relationships
    model_version: Mapped["ModelVersion"] = relationship(back_populates="evaluation_results")


# ─── 10. DatasetConfigurations ────────────────────────────────────────────────
class DatasetConfiguration(Base):
    __tablename__ = "dataset_configurations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    dataset_type: Mapped[str] = mapped_column(String(50), nullable=False)   # "faceforensics" | "celeb_df" | "dfdc" | ...
    base_path: Mapped[str] = mapped_column(String(512), nullable=False)
    train_real_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    train_fake_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    val_real_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    val_fake_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    test_real_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    test_fake_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    num_train_samples: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    num_val_samples: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    num_test_samples: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
