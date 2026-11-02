"""
DeepFake Detection System – Backend
Training Service: background training jobs with stop signal support
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict

from sqlalchemy import select

from app.database.database import AsyncSessionLocal
from app.database.models import ModelVersion, TrainingJob, TrainingMetric

logger = logging.getLogger(__name__)

# Global stop signals dict: job_id -> threading.Event
_stop_signals: Dict[str, threading.Event] = {}


def stop_training_job(job_id: str) -> None:
    """Set the stop signal for a running training job."""
    if job_id in _stop_signals:
        _stop_signals[job_id].set()
        logger.info("Stop signal sent to training job %s", job_id)


async def _update_job(job_id: str, **kwargs: Any) -> None:
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
        job = result.scalar_one_or_none()
        if job:
            for k, v in kwargs.items():
                setattr(job, k, v)
            await db.commit()


async def _save_metric(job_id: str, epoch: int, metrics: Dict[str, Any]) -> None:
    async with AsyncSessionLocal() as db:
        m = TrainingMetric(
            job_id=job_id,
            epoch=epoch,
            train_loss=metrics.get("train_loss", 0.0),
            val_loss=metrics.get("val_loss"),
            train_cls_loss=metrics.get("train_cls_loss"),
            train_mim_loss=metrics.get("train_mim_loss"),
            learning_rate=metrics.get("learning_rate"),
        )
        db.add(m)
        await db.commit()


async def start_pretrain_job(
    job_id: str,
    model_version_id: str,
    config: Dict[str, Any],
) -> None:
    """Background: run self-supervised ViT pretraining."""
    stop_event = threading.Event()
    _stop_signals[job_id] = stop_event

    await _update_job(
        job_id,
        status="running",
        started_at=datetime.now(timezone.utc),
    )

    try:
        from ml.training.pretrain import SelfSupervisedPretrainer

        trainer = SelfSupervisedPretrainer(config=config, stop_event=stop_event)

        async def epoch_callback(epoch: int, metrics: Dict[str, Any]) -> None:
            await _update_job(job_id, current_epoch=epoch)
            await _save_metric(job_id, epoch, metrics)

        checkpoint_path = await trainer.train(epoch_callback=epoch_callback)

        # Mark model as pretrained
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(ModelVersion).where(ModelVersion.id == model_version_id))
            model = result.scalar_one_or_none()
            if model:
                model.pretrain_checkpoint_path = checkpoint_path
                model.training_status = "pretrained"
                await db.commit()

        status = "stopped" if stop_event.is_set() else "completed"
        await _update_job(job_id, status=status, ended_at=datetime.now(timezone.utc))
        logger.info("Pretrain job %s finished with status: %s", job_id, status)

    except ImportError as e:
        await _update_job(
            job_id,
            status="failed",
            error_message=f"ML dependencies not installed: {e}",
            ended_at=datetime.now(timezone.utc),
        )
    except Exception as e:
        logger.exception("Pretrain job %s failed", job_id)
        await _update_job(
            job_id,
            status="failed",
            error_message=str(e),
            ended_at=datetime.now(timezone.utc),
        )
    finally:
        _stop_signals.pop(job_id, None)


async def start_classifier_job(
    job_id: str,
    model_version_id: str,
    config: Dict[str, Any],
) -> None:
    """Background: train the supervised Graph Transformer classifier."""
    stop_event = threading.Event()
    _stop_signals[job_id] = stop_event

    await _update_job(
        job_id,
        status="running",
        started_at=datetime.now(timezone.utc),
    )

    try:
        from ml.training.classifier_train import GraphTransformerTrainer

        trainer = GraphTransformerTrainer(config=config, stop_event=stop_event)

        best_val_loss = None

        async def epoch_callback(epoch: int, metrics: Dict[str, Any]) -> None:
            nonlocal best_val_loss
            val_loss = metrics.get("val_loss")
            if val_loss is not None:
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
            await _update_job(job_id, current_epoch=epoch, best_val_loss=best_val_loss)
            await _save_metric(job_id, epoch, metrics)

        checkpoint_path = await trainer.train(epoch_callback=epoch_callback)

        async with AsyncSessionLocal() as db:
            result = await db.execute(select(ModelVersion).where(ModelVersion.id == model_version_id))
            model = result.scalar_one_or_none()
            if model:
                model.checkpoint_path = checkpoint_path
                model.training_status = "trained"
                model.is_active = True
                await db.commit()

        status = "stopped" if stop_event.is_set() else "completed"
        await _update_job(job_id, status=status, ended_at=datetime.now(timezone.utc))
        logger.info("Classifier training %s finished: %s", job_id, status)

    except ImportError as e:
        await _update_job(
            job_id,
            status="failed",
            error_message=f"ML dependencies not installed: {e}",
            ended_at=datetime.now(timezone.utc),
        )
    except Exception as e:
        logger.exception("Classifier training job %s failed", job_id)
        await _update_job(
            job_id,
            status="failed",
            error_message=str(e),
            ended_at=datetime.now(timezone.utc),
        )
    finally:
        _stop_signals.pop(job_id, None)
