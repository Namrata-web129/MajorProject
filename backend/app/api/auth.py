"""
DeepFake Detection System – Backend
Authentication API Router
"""
from __future__ import annotations

import logging
from datetime import timedelta

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.core.config import get_settings
from app.core.dependencies import CurrentUser, DBSession
from app.core.security import create_access_token, hash_password, verify_password
from app.database.models import User
from app.schemas.auth import (
    LoginRequest,
    MessageResponse,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest, db: DBSession) -> TokenResponse:
    """Register a new user account."""
    # Check email uniqueness
    result = await db.execute(select(User).where(User.email == payload.email))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists",
        )
    # Check username uniqueness
    result = await db.execute(select(User).where(User.username == payload.username))
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username is already taken",
        )

    user = User(
        email=payload.email,
        username=payload.username,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
        role="user",
    )
    db.add(user)
    await db.flush()   # get id before commit
    await db.refresh(user)

    token = create_access_token({"sub": user.id, "role": user.role})
    logger.info("New user registered: %s", user.email)
    return TokenResponse(access_token=token, user=UserResponse.model_validate(user))


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: DBSession) -> TokenResponse:
    """Authenticate a user and return a JWT token."""
    result = await db.execute(select(User).where(User.email == payload.email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )

    token = create_access_token(
        {"sub": user.id, "role": user.role},
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    logger.info("User logged in: %s", user.email)
    return TokenResponse(access_token=token, user=UserResponse.model_validate(user))


@router.post("/logout", response_model=MessageResponse)
async def logout(current_user: CurrentUser) -> MessageResponse:
    """Logout (client should discard token; stateless JWT)."""
    logger.info("User logged out: %s", current_user.email)
    return MessageResponse(message="Logged out successfully")


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: CurrentUser) -> UserResponse:
    """Return the currently authenticated user's profile."""
    return UserResponse.model_validate(current_user)
