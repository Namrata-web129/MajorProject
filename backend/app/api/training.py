"""
DeepFake Detection System – Backend
Training API Router (pretrain + classifier training, admin only)
"""
from __future__ import annotations

import json
import logging
from typing import List

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from sqlalchemy import desc, select

from app.core.dependencies import AdminUser, DBSession
from app.database.models import ModelVersion, TrainingJob, TrainingMetric
from app.schemas.schemas import (
    ClassifierTrainConfig,
    PretrainConfig,
    TrainingJobResponse,
    TrainingStatusResponse,
)
from app.services.training_service import start_pretrain_job, start_classifier_job, stop_training_job

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/training", tags=["Training"])


@router.post("/pretrain", response_model=TrainingJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_pretrain(
    config: PretrainConfig,
    background_tasks: BackgroundTasks,
    admin: AdminUser,
    db: DBSession,
) -> TrainingJobResponse:
    """Start self-supervised ViT pretraining (admin only)."""
    # Create/get model version record
    model = ModelVersion(
        name=config.model_name,
        training_status="pretraining",
    )
    db.add(model)
    await db.flush()

    job = TrainingJob(
        model_version_id=model.id,
        job_type="pretrain",
        status="queued",
        config=json.dumps(config.model_dump()),
        dataset_path=config.dataset_path,
        total_epochs=config.epochs,
    )
    db.add(job)
    await db.flush()
    job_id = job.id
    model_id = model.id
    await db.commit()

    background_tasks.add_task(
        start_pretrain_job,
        job_id=job_id,
        model_version_id=model_id,
        config=config.model_dump(),
    )

    return TrainingJobResponse(
        job_id=job_id,
        status="queued",
        message="Pretraining job queued successfully",
    )


@router.post("/classifier", response_model=TrainingJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_classifier(
    config: ClassifierTrainConfig,
    background_tasks: BackgroundTasks,
    admin: AdminUser,
    db: DBSession,
) -> TrainingJobResponse:
    """Start supervised Graph Transformer classifier training (admin only)."""
    model = ModelVersion(
        name=config.model_name,
        training_status="training",
    )
    db.add(model)
    await db.flush()

    job = TrainingJob(
        model_version_id=model.id,
        job_type="classifier",
        status="queued",
        config=json.dumps(config.model_dump()),
        dataset_path=config.dataset_path,
        total_epochs=config.epochs,
    )
    db.add(job)
    await db.flush()
    job_id = job.id
    model_id = model.id
    await db.commit()

    background_tasks.add_task(
        start_classifier_job,
        job_id=job_id,
        model_version_id=model_id,
        config=config.model_dump(),
    )

    return TrainingJobResponse(
        job_id=job_id,
        status="queued",
        message="Classifier training job queued successfully",
    )


@router.get("/status/{job_id}", response_model=TrainingStatusResponse)
async def get_training_status(
    job_id: str,
    admin: AdminUser,
    db: DBSession,
) -> TrainingStatusResponse:
    """Get real-time training job status and metrics."""
    result = await db.execute(
        select(TrainingJob)
        .where(TrainingJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    metrics_result = await db.execute(
        select(TrainingMetric)
        .where(TrainingMetric.job_id == job_id)
        .order_by(TrainingMetric.epoch)
    )
    metrics = metrics_result.scalars().all()

    progress = (job.current_epoch / job.total_epochs * 100) if job.total_epochs > 0 else 0

    return TrainingStatusResponse(
        job_id=job.id,
        job_type=job.job_type,
        status=job.status,
        current_epoch=job.current_epoch,
        total_epochs=job.total_epochs,
        progress_pct=round(progress, 1),
        best_val_loss=job.best_val_loss,
        started_at=job.started_at,
        metrics=[
            {
                "epoch": m.epoch,
                "train_loss": m.train_loss,
                "val_loss": m.val_loss,
                "train_cls_loss": m.train_cls_loss,
                "train_mim_loss": m.train_mim_loss,
                "learning_rate": m.learning_rate,
            }
            for m in metrics
        ],
    )


@router.post("/stop/{job_id}", response_model=TrainingJobResponse)
async def stop_training(
    job_id: str,
    admin: AdminUser,
    db: DBSession,
) -> TrainingJobResponse:
    """Request a running training job to stop gracefully."""
    result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    stop_training_job(job_id)
    return TrainingJobResponse(job_id=job_id, status="stopping", message="Stop signal sent")


@router.get("/history", response_model=List[TrainingStatusResponse])
async def get_training_history(
    admin: AdminUser,
    db: DBSession,
    limit: int = 20,
) -> List[TrainingStatusResponse]:
    """Get recent training job history."""
    result = await db.execute(
        select(TrainingJob).order_by(desc(TrainingJob.created_at)).limit(limit)
    )
    jobs = result.scalars().all()
    return [
        TrainingStatusResponse(
            job_id=j.id,
            job_type=j.job_type,
            status=j.status,
            current_epoch=j.current_epoch,
            total_epochs=j.total_epochs,
            progress_pct=round((j.current_epoch / j.total_epochs * 100) if j.total_epochs > 0 else 0, 1),
            best_val_loss=j.best_val_loss,
            started_at=j.started_at,
            metrics=[],
        )
        for j in jobs
    ]
