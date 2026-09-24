"""
schemas/user.py — Pydantic schemas cho User & Auth.
"""
import uuid
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


# ── Request ───────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=100)
    password: str = Field(min_length=8)
    full_name: str | None = None


class UserUpdate(BaseModel):
    full_name: str | None = None
    password: str | None = Field(default=None, min_length=8)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


# ── Response ──────────────────────────────────────────────────────────────────

class UserRead(BaseModel):
    id: uuid.UUID
    email: str
    username: str
    full_name: str | None
    is_active: bool
    is_admin: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ── JWT ───────────────────────────────────────────────────────────────────────

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: uuid.UUID | None = None
