"""
DeepFake Detection System – Backend
Evaluation API Router
"""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from sqlalchemy import desc, select

from app.core.dependencies import AdminUser, CurrentUser, DBSession
from app.database.models import EvaluationResult
from app.schemas.schemas import EvaluationRequest, EvaluationResultSchema
from app.services.evaluation_service import run_evaluation_job

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/evaluation", tags=["Evaluation"])


@router.post("/run", status_code=status.HTTP_202_ACCEPTED)
async def run_evaluation(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks,
    admin: AdminUser,
    db: DBSession,
) -> dict:
    """Trigger a model evaluation run (admin only)."""
    background_tasks.add_task(
        run_evaluation_job,
        model_version_id=request.model_version_id,
        eval_type=request.eval_type,
        dataset_path=request.dataset_path,
        dataset_name=request.dataset_name,
    )
    return {"message": "Evaluation job started", "status": "queued"}


@router.get("/results", response_model=List[EvaluationResultSchema])
async def get_evaluation_results(
    current_user: CurrentUser,
    db: DBSession,
    limit: int = 50,
) -> List[EvaluationResultSchema]:
    """List recent evaluation results."""
    result = await db.execute(
        select(EvaluationResult)
        .order_by(desc(EvaluationResult.evaluated_at))
        .limit(limit)
    )
    return [EvaluationResultSchema.model_validate(r) for r in result.scalars().all()]


@router.get("/metrics/{result_id}")
async def get_evaluation_metrics(
    result_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> dict:
    """Return full metrics for a specific evaluation result, including ROC and confusion matrix data."""
    result = await db.execute(
        select(EvaluationResult).where(EvaluationResult.id == result_id)
    )
    ev = result.scalar_one_or_none()
    if not ev:
        raise HTTPException(status_code=404, detail="Evaluation result not found")

    import json
    return {
        "id": ev.id,
        "eval_type": ev.eval_type,
        "dataset_name": ev.dataset_name,
        "accuracy": ev.accuracy,
        "precision": ev.precision,
        "recall": ev.recall,
        "f1_score": ev.f1_score,
        "auc_roc": ev.auc_roc,
        "false_positive_rate": ev.false_positive_rate,
        "false_negative_rate": ev.false_negative_rate,
        "confusion_matrix": json.loads(ev.confusion_matrix) if ev.confusion_matrix else None,
        "roc_curve_data": json.loads(ev.roc_curve_data) if ev.roc_curve_data else None,
        "extra_metrics": json.loads(ev.extra_metrics) if ev.extra_metrics else None,
        "evaluated_at": ev.evaluated_at.isoformat(),
    }
