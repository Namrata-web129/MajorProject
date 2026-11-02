"""
DeepFake Detection System – Backend
Model API Router
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.core.config import get_settings
from app.core.dependencies import AdminUser, CurrentUser, DBSession
from app.database.models import ModelVersion
from app.schemas.schemas import ModelInfoResponse, ModelStatusResponse

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/model", tags=["Model"])


@router.get("/info", response_model=ModelInfoResponse)
async def get_model_info(current_user: CurrentUser, db: DBSession) -> ModelInfoResponse:
    """Return information about the currently active model version."""
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.is_active == True).limit(1)  # noqa: E712
    )
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="No active model version found")
    return ModelInfoResponse.model_validate(model)


@router.get("/status", response_model=ModelStatusResponse)
async def get_model_status(current_user: CurrentUser, db: DBSession) -> ModelStatusResponse:
    """Return the current model loading and training status."""
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.is_active == True).limit(1)  # noqa: E712
    )
    model = result.scalar_one_or_none()

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "unknown"

    checkpoint_exists = False
    pretrain_checkpoint_exists = False
    if model:
        if model.checkpoint_path:
            checkpoint_exists = Path(model.checkpoint_path).exists()
        if model.pretrain_checkpoint_path:
            pretrain_checkpoint_exists = Path(model.pretrain_checkpoint_path).exists()

    return ModelStatusResponse(
        model_loaded=checkpoint_exists,
        training_status=model.training_status if model else "untrained",
        device=device,
        checkpoint_exists=checkpoint_exists,
        pretrain_checkpoint_exists=pretrain_checkpoint_exists,
        active_model_id=model.id if model else None,
    )


@router.get("/checkpoints")
async def list_checkpoints(current_user: AdminUser) -> List[str]:
    """List checkpoint files available on disk (admin only)."""
    cp_dir = Path(settings.CHECKPOINTS_DIR)
    if not cp_dir.exists():
        return []
    return [str(p.name) for p in cp_dir.glob("*.pt")] + [str(p.name) for p in cp_dir.glob("*.pth")]
