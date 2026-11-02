"""
DeepFake Detection System – Backend
Detection Service: orchestrates ML inference for image and video
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.database.database import AsyncSessionLocal
from app.database.models import AnalysisHistory, DetectionResult, VideoFrameResult

logger = logging.getLogger(__name__)
settings = get_settings()


async def _update_analysis_status(
    db: AsyncSession,
    analysis_id: str,
    status: str,
    error_message: Optional[str] = None,
    processing_duration_ms: Optional[int] = None,
) -> None:
    result = await db.execute(select(AnalysisHistory).where(AnalysisHistory.id == analysis_id))
    analysis = result.scalar_one_or_none()
    if analysis:
        analysis.status = status
        if error_message:
            analysis.error_message = error_message
        if processing_duration_ms is not None:
            analysis.processing_duration_ms = processing_duration_ms
        analysis.updated_at = datetime.now(timezone.utc)
        await db.commit()


async def run_image_detection(analysis_id: str, image_path: str) -> None:
    """
    Background task: run the full image detection pipeline.
    Uses its own DB session since this runs outside the request context.
    """
    start_time = time.time()
    logger.info("Starting image detection for analysis %s", analysis_id)

    async with AsyncSessionLocal() as db:
        try:
            # ── Import ML pipeline (lazy import to avoid startup delay) ─────
            from ml.inference.pipeline import InferencePipeline

            pipeline = InferencePipeline()
            result = await pipeline.analyze_image(image_path)

            elapsed_ms = int((time.time() - start_time) * 1000)

            # Persist result
            det = DetectionResult(
                analysis_id=analysis_id,
                prediction=result["prediction"],
                real_probability=result["real_probability"],
                fake_probability=result["fake_probability"],
                confidence=result["confidence"],
                raw_logits=json.dumps(result.get("raw_logits", [])),
                face_count=result.get("face_count", 1),
                face_crops_paths=json.dumps(result.get("face_crops_paths", [])),
                heatmap_path=result.get("heatmap_path"),
                graph_data_path=result.get("graph_data_path"),
                patch_relevance=json.dumps(result.get("patch_relevance", [])),
            )
            db.add(det)

            await _update_analysis_status(
                db, analysis_id, "completed", processing_duration_ms=elapsed_ms
            )
            logger.info(
                "Image detection completed for %s: %s (%.1f%% fake confidence)",
                analysis_id,
                result["prediction"],
                result["fake_probability"] * 100,
            )

        except ImportError as e:
            logger.warning("ML modules not yet available: %s", e)
            await _update_analysis_status(
                db,
                analysis_id,
                "failed",
                error_message=(
                    "Model not loaded: ML dependencies not installed or model not trained. "
                    "Please install requirements and train the model before running inference. "
                    f"Details: {str(e)}"
                ),
            )
        except Exception as e:
            logger.exception("Image detection failed for analysis %s", analysis_id)
            await _update_analysis_status(
                db, analysis_id, "failed", error_message=str(e)
            )


async def run_video_detection(analysis_id: str, video_path: str) -> None:
    """
    Background task: extract frames, run per-frame detection, aggregate results.
    """
    start_time = time.time()
    logger.info("Starting video detection for analysis %s", analysis_id)

    async with AsyncSessionLocal() as db:
        try:
            from ml.inference.pipeline import InferencePipeline

            pipeline = InferencePipeline()
            video_result = await pipeline.analyze_video(video_path)

            elapsed_ms = int((time.time() - start_time) * 1000)

            # Store aggregated result
            agg = video_result["aggregated"]
            det = DetectionResult(
                analysis_id=analysis_id,
                prediction=agg["prediction"],
                real_probability=agg["real_probability"],
                fake_probability=agg["fake_probability"],
                confidence=agg["confidence"],
                face_count=agg.get("face_count", 0),
            )
            db.add(det)

            # Store per-frame results
            for frame in video_result.get("frames", []):
                fv = VideoFrameResult(
                    analysis_id=analysis_id,
                    frame_index=frame["frame_index"],
                    timestamp_seconds=frame["timestamp_seconds"],
                    prediction=frame["prediction"],
                    real_probability=frame["real_probability"],
                    fake_probability=frame["fake_probability"],
                    face_detected=frame.get("face_detected", True),
                )
                db.add(fv)

            await _update_analysis_status(
                db, analysis_id, "completed", processing_duration_ms=elapsed_ms
            )
            logger.info(
                "Video detection completed for %s: %s",
                analysis_id,
                agg["prediction"],
            )

        except ImportError as e:
            logger.warning("ML modules not available: %s", e)
            await _update_analysis_status(
                db,
                analysis_id,
                "failed",
                error_message=(
                    "Model not loaded: ML dependencies not installed or model not trained. "
                    f"Details: {str(e)}"
                ),
            )
        except Exception as e:
            logger.exception("Video detection failed for analysis %s", analysis_id)
            await _update_analysis_status(
                db, analysis_id, "failed", error_message=str(e)
            )
