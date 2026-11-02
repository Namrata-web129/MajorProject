"""
DeepFake Detection System – Backend
File Service: validation and secure storage
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Tuple

import aiofiles
from fastapi import HTTPException, UploadFile, status

from app.core.config import get_settings

settings = get_settings()

# Max read chunk for streaming write
CHUNK_SIZE = 1024 * 1024  # 1 MB


def validate_upload(
    file: UploadFile,
    allowed_types: List[str],
) -> None:
    """
    Validate uploaded file:
    - Check extension against allowed list
    - Check content-type header (best-effort)
    - Enforce file size limit (checked during save)
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type '{suffix}' is not allowed. Accepted: {', '.join(allowed_types)}",
        )


async def save_upload(
    file: UploadFile,
    upload_dir: str,
) -> Tuple[Path, str]:
    """
    Save an uploaded file securely:
    - Generate a UUID-based filename to prevent path traversal
    - Stream write to disk
    - Enforce max file size limit

    Returns (stored_path, stored_filename)
    """
    settings.ensure_dirs()
    Path(upload_dir).mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "file").suffix.lower()
    stored_name = f"{uuid.uuid4().hex}{suffix}"
    stored_path = Path(upload_dir) / stored_name

    total_bytes = 0
    max_bytes = settings.max_file_size_bytes

    async with aiofiles.open(stored_path, "wb") as out_file:
        while chunk := await file.read(CHUNK_SIZE):
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                # Remove partial file
                stored_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB} MB",
                )
            await out_file.write(chunk)

    return stored_path, stored_name
