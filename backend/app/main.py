"""
DeepFake Detection System – Backend
FastAPI Application Entry Point
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.database.database import init_db
from app.database.models import User  # noqa: F401 — registers model with Base

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

settings = get_settings()


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("═══ DeepFake Detection System starting up ═══")
    settings.ensure_dirs()
    await init_db()
    await _seed_admin()
    await _seed_model_version()
    logger.info("Startup complete. Device selection: %s", settings.DEVICE)
    yield
    logger.info("═══ DeepFake Detection System shutting down ═══")


async def _seed_admin() -> None:
    """Create the default admin account if it doesn't exist."""
    from sqlalchemy import select
    from app.core.security import hash_password
    from app.database.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(User).where(User.email == settings.FIRST_ADMIN_EMAIL)
        )
        if not result.scalar_one_or_none():
            admin = User(
                email=settings.FIRST_ADMIN_EMAIL,
                username="admin",
                hashed_password=hash_password(settings.FIRST_ADMIN_PASSWORD),
                full_name="System Administrator",
                role="admin",
            )
            db.add(admin)
            await db.commit()
            logger.info("Admin account seeded: %s", settings.FIRST_ADMIN_EMAIL)


async def _seed_model_version() -> None:
    """Create a default model version record if none exists."""
    from sqlalchemy import select
    from app.database.database import AsyncSessionLocal
    from app.database.models import ModelVersion

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(ModelVersion).limit(1))
        if not result.scalar_one_or_none():
            model = ModelVersion(
                name="SelfSupervisedGraphTransformer_v1",
                description=(
                    "Self-Supervised Graph Transformer for Deepfake Detection "
                    "(Khormali & Yuan, IEEE Access 2024)"
                ),
                training_status="untrained",
                is_active=False,
            )
            db.add(model)
            await db.commit()
            logger.info("Default model version record created")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "A research-oriented deepfake detection platform based on the Self-Supervised "
        "Graph Transformer architecture (Khormali & Yuan, IEEE Access 2024)."
    ),
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global Exception Handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please try again later."},
    )


# ── Routers ───────────────────────────────────────────────────────────────────
from app.api.auth import router as auth_router
from app.api.detection import router as detection_router
from app.api.evaluation import router as evaluation_router
from app.api.explainability import router as explainability_router
from app.api.model import router as model_router
from app.api.training import router as training_router

app.include_router(auth_router)
app.include_router(detection_router)
app.include_router(model_router)
app.include_router(training_router)
app.include_router(evaluation_router)
app.include_router(explainability_router)


@app.get("/api/health")
async def health_check() -> dict:
    return {"status": "healthy", "version": settings.APP_VERSION}
