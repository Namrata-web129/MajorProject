"""
DeepFake Detection System – Backend
Evaluation Service (background task)
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


async def run_evaluation_job(
    model_version_id: str,
    eval_type: str,
    dataset_path: str,
    dataset_name: str,
) -> None:
    """
    Background task: run model evaluation and store results.
    """
    logger.info(
        "Starting evaluation job: model=%s, type=%s, dataset=%s",
        model_version_id, eval_type, dataset_name,
    )

    from app.database.database import AsyncSessionLocal
    from app.database.models import EvaluationResult

    try:
        from ml.evaluation.metrics import ModelEvaluator

        evaluator = ModelEvaluator(model_version_id=model_version_id)
        metrics: Dict[str, Any] = await evaluator.evaluate(
            dataset_path=dataset_path,
            eval_type=eval_type,
        )

        async with AsyncSessionLocal() as db:
            ev = EvaluationResult(
                model_version_id=model_version_id,
                eval_type=eval_type,
                dataset_name=dataset_name,
                accuracy=metrics.get("accuracy"),
                precision=metrics.get("precision"),
                recall=metrics.get("recall"),
                f1_score=metrics.get("f1_score"),
                auc_roc=metrics.get("auc_roc"),
                false_positive_rate=metrics.get("false_positive_rate"),
                false_negative_rate=metrics.get("false_negative_rate"),
                confusion_matrix=json.dumps(metrics.get("confusion_matrix", [])),
                roc_curve_data=json.dumps(metrics.get("roc_curve_data", {})),
                extra_metrics=json.dumps(metrics.get("extra_metrics", {})),
            )
            db.add(ev)
            await db.commit()
        logger.info("Evaluation completed for model %s", model_version_id)

    except ImportError as e:
        logger.warning("Evaluation skipped – ML dependencies missing: %s", e)
    except Exception as e:
        logger.exception("Evaluation failed: %s", e)
