"""
DeepFake Detection System – Backend
Explainability API Router
"""
from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.dependencies import CurrentUser, DBSession
from app.database.models import AnalysisHistory, DetectionResult
from app.schemas.schemas import ExplainabilityResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/explainability", tags=["Explainability"])


@router.get("/{analysis_id}", response_model=ExplainabilityResponse)
async def get_explainability(
    analysis_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> ExplainabilityResponse:
    """Return explainability data for an analysis: heatmap, patch relevance, attention."""
    result = await db.execute(
        select(AnalysisHistory)
        .options(selectinload(AnalysisHistory.detection_result))
        .where(AnalysisHistory.id == analysis_id, AnalysisHistory.user_id == current_user.id)
    )
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    det: DetectionResult | None = analysis.detection_result
    if not det:
        raise HTTPException(status_code=404, detail="Detection result not yet available")

    patch_relevance = None
    if det.patch_relevance:
        try:
            patch_relevance = json.loads(det.patch_relevance)
        except json.JSONDecodeError:
            pass

    heatmap_b64 = None
    if det.heatmap_path and Path(det.heatmap_path).exists():
        with open(det.heatmap_path, "rb") as f:
            heatmap_b64 = base64.b64encode(f.read()).decode("utf-8")

    explanation = (
        f"The model classified this image as {det.prediction} with "
        f"{det.confidence * 100:.1f}% confidence. "
        "The heatmap highlights facial regions that most influenced the decision. "
        "Brighter areas indicate patches with higher relevance scores from the "
        "Graph Transformer attention mechanism."
    )

    return ExplainabilityResponse(
        analysis_id=analysis_id,
        prediction=det.prediction,
        confidence=det.confidence,
        heatmap_base64=heatmap_b64,
        patch_relevance=patch_relevance,
        explanation_text=explanation,
    )


@router.get("/{analysis_id}/heatmap")
async def get_heatmap(
    analysis_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> dict:
    """Return only the base64-encoded heatmap image."""
    result = await db.execute(
        select(DetectionResult)
        .join(AnalysisHistory)
        .where(
            DetectionResult.analysis_id == analysis_id,
            AnalysisHistory.user_id == current_user.id,
        )
    )
    det = result.scalar_one_or_none()
    if not det:
        raise HTTPException(status_code=404, detail="Detection result not found")
    if not det.heatmap_path or not Path(det.heatmap_path).exists():
        raise HTTPException(status_code=404, detail="Heatmap not available for this analysis")

    with open(det.heatmap_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return {"heatmap_base64": b64, "format": "jpeg"}


@router.get("/{analysis_id}/graph")
async def get_graph_data(
    analysis_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> dict:
    """Return graph node relevance data for visualization."""
    result = await db.execute(
        select(DetectionResult)
        .join(AnalysisHistory)
        .where(
            DetectionResult.analysis_id == analysis_id,
            AnalysisHistory.user_id == current_user.id,
        )
    )
    det = result.scalar_one_or_none()
    if not det:
        raise HTTPException(status_code=404, detail="Detection result not found")

    graph_data = None
    if det.graph_data_path and Path(det.graph_data_path).exists():
        with open(det.graph_data_path, "r") as f:
            graph_data = json.load(f)

    patch_relevance = None
    if det.patch_relevance:
        try:
            patch_relevance = json.loads(det.patch_relevance)
        except json.JSONDecodeError:
            pass

    return {
        "analysis_id": analysis_id,
        "patch_relevance": patch_relevance,
        "graph_data": graph_data,
        "num_patches": 256,
        "grid_size": 16,
    }
