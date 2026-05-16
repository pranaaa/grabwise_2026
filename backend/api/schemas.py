"""Pydantic schemas for the GrabWise HTTP API."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

UserRole = Literal["driver", "customer", "merchant", "admin"]


class ChatRequest(BaseModel):
    """Authenticated chat request — role/user_id come from the session, not the body."""
    message: str = Field(..., min_length=1, max_length=2000)
    thread_id: str | None = Field(default=None)
    # Optional Bedrock model_id (must be in backend.llm.registry.BEDROCK_MODELS).
    # If unset / unknown, the factory falls back to BEDROCK_MODEL_SONNET/HAIKU env defaults.
    model: str | None = Field(default=None, max_length=120)


class UserSummary(BaseModel):
    id: int
    name: str
    role: UserRole
    city: str
    extra: str | None = None
