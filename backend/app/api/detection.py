"""
DeepFake Detection System – Backend
Detection API Router (image + video upload and analysis)
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, status
from sqlalchemy import select

from app.core.config import get_settings
from app.core.dependencies import CurrentUser, DBSession
from app.database.models import AnalysisHistory, DetectionResult, UploadedFile
from app.schemas.schemas import (
    AnalysisSummary,
    FullAnalysisResponse,
    PaginatedHistory,
)
from app.services.detection_service import run_image_detection, run_video_detection
from app.services.file_service import save_upload, validate_upload

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/detection", tags=["Detection"])


# ── Helper ─────────────────────────────────────────────────────────────────────
async def _get_analysis_or_404(analysis_id: str, user_id: str, db: DBSession) -> AnalysisHistory:
    result = await db.execute(
        select(AnalysisHistory)
        .where(AnalysisHistory.id == analysis_id, AnalysisHistory.user_id == user_id)
    )
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    return analysis


# ── Image Detection ────────────────────────────────────────────────────────────
@router.post("/image", response_model=FullAnalysisResponse, status_code=status.HTTP_202_ACCEPTED)
async def analyze_image(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    current_user: CurrentUser,
    db: DBSession,
) -> FullAnalysisResponse:
    """Upload an image and run deepfake detection."""
    # Validate
    validate_upload(file, allowed_types=settings.allowed_image_extensions)

    # Save file
    stored_path, stored_name = await save_upload(file, settings.UPLOAD_DIR)

    # Create DB records
    uploaded_file = UploadedFile(
        user_id=current_user.id,
        original_filename=file.filename or "unknown",
        stored_filename=stored_name,
        file_path=str(stored_path),
        file_type="image",
        mime_type=file.content_type,
        file_size=stored_path.stat().st_size,
    )
    db.add(uploaded_file)
    await db.flush()

    analysis = AnalysisHistory(
        user_id=current_user.id,
        file_id=uploaded_file.id,
        status="processing",
    )
    db.add(analysis)
    await db.flush()
    analysis_id = analysis.id
    await db.commit()

    # Queue background detection
    background_tasks.add_task(
        run_image_detection,
        analysis_id=analysis_id,
        image_path=str(stored_path),
    )

    return FullAnalysisResponse(
        analysis=AnalysisSummary(
            id=analysis_id,
            status="processing",
            file_name=file.filename,
            file_type="image",
            created_at=analysis.created_at,
        )
    )


# ── Video Detection ────────────────────────────────────────────────────────────
@router.post("/video", response_model=FullAnalysisResponse, status_code=status.HTTP_202_ACCEPTED)
async def analyze_video(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    current_user: CurrentUser,
    db: DBSession,
) -> FullAnalysisResponse:
    """Upload a video and run frame-level deepfake detection."""
    validate_upload(file, allowed_types=settings.allowed_video_extensions)

    stored_path, stored_name = await save_upload(file, settings.UPLOAD_DIR)

    uploaded_file = UploadedFile(
        user_id=current_user.id,
        original_filename=file.filename or "unknown",
        stored_filename=stored_name,
        file_path=str(stored_path),
        file_type="video",
        mime_type=file.content_type,
        file_size=stored_path.stat().st_size,
    )
    db.add(uploaded_file)
    await db.flush()

    analysis = AnalysisHistory(
        user_id=current_user.id,
        file_id=uploaded_file.id,
        status="processing",
    )
    db.add(analysis)
    await db.flush()
    analysis_id = analysis.id
    await db.commit()

    background_tasks.add_task(
        run_video_detection,
        analysis_id=analysis_id,
        video_path=str(stored_path),
    )

    return FullAnalysisResponse(
        analysis=AnalysisSummary(
            id=analysis_id,
            status="processing",
            file_name=file.filename,
            file_type="video",
            created_at=analysis.created_at,
        )
    )


# ── Get Analysis Result ────────────────────────────────────────────────────────
@router.get("/{analysis_id}", response_model=FullAnalysisResponse)
async def get_analysis(
    analysis_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> FullAnalysisResponse:
    """Retrieve a specific analysis result (owner only)."""
    from sqlalchemy.orm import selectinload

    result = await db.execute(
        select(AnalysisHistory)
        .options(
            selectinload(AnalysisHistory.detection_result),
            selectinload(AnalysisHistory.video_frame_results),
            selectinload(AnalysisHistory.file),
        )
        .where(AnalysisHistory.id == analysis_id, AnalysisHistory.user_id == current_user.id)
    )
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    det = analysis.detection_result
    frames = analysis.video_frame_results

    suspicious = None
    if frames:
        threshold = settings.CONFIDENCE_THRESHOLD
        suspicious = [
            f.timestamp_seconds for f in frames
            if f.prediction == "FAKE" and f.fake_probability >= threshold
        ]

    return FullAnalysisResponse(
        analysis=AnalysisSummary(
            id=analysis.id,
            status=analysis.status,
            prediction=det.prediction if det else None,
            real_probability=det.real_probability if det else None,
            fake_probability=det.fake_probability if det else None,
            confidence=det.confidence if det else None,
            face_count=det.face_count if det else None,
            file_name=analysis.file.original_filename if analysis.file else None,
            file_type=analysis.file.file_type if analysis.file else None,
            created_at=analysis.created_at,
            processing_duration_ms=analysis.processing_duration_ms,
        ),
        detection=None if not det else None,  # serialized separately below
        video_frames=None if not frames else [
            {
                "frame_index": f.frame_index,
                "timestamp_seconds": f.timestamp_seconds,
                "prediction": f.prediction,
                "real_probability": f.real_probability,
                "fake_probability": f.fake_probability,
                "face_detected": f.face_detected,
            }
            for f in frames
        ],
        suspicious_timestamps=suspicious,
    )


# ── History ────────────────────────────────────────────────────────────────────
@router.get("/history/list", response_model=PaginatedHistory)
async def get_history(
    current_user: CurrentUser,
    db: DBSession,
    page: int = 1,
    page_size: int = 20,
) -> PaginatedHistory:
    """Return paginated analysis history for the current user."""
    from sqlalchemy import func, desc
    from sqlalchemy.orm import selectinload

    offset = (page - 1) * page_size

    count_result = await db.execute(
        select(func.count()).select_from(AnalysisHistory)
        .where(AnalysisHistory.user_id == current_user.id)
    )
    total = count_result.scalar_one()

    result = await db.execute(
        select(AnalysisHistory)
        .options(
            selectinload(AnalysisHistory.detection_result),
            selectinload(AnalysisHistory.file),
        )
        .where(AnalysisHistory.user_id == current_user.id)
        .order_by(desc(AnalysisHistory.created_at))
        .offset(offset)
        .limit(page_size)
    )
    analyses = result.scalars().all()

    items = []
    for a in analyses:
        det = a.detection_result
        items.append(AnalysisSummary(
            id=a.id,
            status=a.status,
            prediction=det.prediction if det else None,
            real_probability=det.real_probability if det else None,
            fake_probability=det.fake_probability if det else None,
            confidence=det.confidence if det else None,
            face_count=det.face_count if det else None,
            file_name=a.file.original_filename if a.file else None,
            file_type=a.file.file_type if a.file else None,
            created_at=a.created_at,
            processing_duration_ms=a.processing_duration_ms,
        ))

    return PaginatedHistory(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        has_next=(offset + page_size) < total,
    )


# ── Delete Analysis ────────────────────────────────────────────────────────────
@router.delete("/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_analysis(
    analysis_id: str,
    current_user: CurrentUser,
    db: DBSession,
) -> None:
    """Delete an analysis record (owner only)."""
    analysis = await _get_analysis_or_404(analysis_id, current_user.id, db)
    await db.delete(analysis)
    await db.commit()
