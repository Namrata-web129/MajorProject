"""
DeepFake Detection System – Backend
Core Configuration (Pydantic Settings v2)
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    APP_NAME: str = "DeepFake Detection System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./deepfake.db"

    # ── JWT ───────────────────────────────────────────────────────────────────
    SECRET_KEY: str = "CHANGE_ME_to_a_64_char_random_hex_string_before_production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # ── Admin seed ────────────────────────────────────────────────────────────
    FIRST_ADMIN_EMAIL: str = "admin@deepfake.ai"
    FIRST_ADMIN_PASSWORD: str = "Admin@123"

    # ── File Storage ──────────────────────────────────────────────────────────
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE_MB: int = 500
    ALLOWED_IMAGE_TYPES: str = "jpg,jpeg,png"
    ALLOWED_VIDEO_TYPES: str = "mp4,avi,mov"

    # ── ML Model ──────────────────────────────────────────────────────────────
    CHECKPOINTS_DIR: str = "./checkpoints"
    MODEL_EMBED_DIM: int = 256
    MODEL_PATCH_SIZE: int = 20
    MODEL_IMAGE_SIZE: int = 320
    MODEL_GCN_LAYERS: int = 3
    MODEL_TRANSFORMER_BLOCKS: int = 3
    MODEL_ATTENTION_HEADS: int = 8
    MODEL_KNN_K: int = 8
    CONFIDENCE_THRESHOLD: float = 0.5
    DEVICE: str = "auto"

    # ── Video ─────────────────────────────────────────────────────────────────
    VIDEO_FRAME_SAMPLE_RATE: int = 1
    VIDEO_MAX_FRAMES: int = 50
    VIDEO_PROCESSING_TIMEOUT: int = 300

    # ── CORS ──────────────────────────────────────────────────────────────────
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60

    # ── Computed properties ───────────────────────────────────────────────────
    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]

    @property
    def allowed_image_extensions(self) -> List[str]:
        return [f".{e.strip()}" for e in self.ALLOWED_IMAGE_TYPES.split(",")]

    @property
    def allowed_video_extensions(self) -> List[str]:
        return [f".{e.strip()}" for e in self.ALLOWED_VIDEO_TYPES.split(",")]

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        for d in [self.UPLOAD_DIR, self.CHECKPOINTS_DIR, "./reports", "./experiments"]:
            Path(d).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    return Settings()
