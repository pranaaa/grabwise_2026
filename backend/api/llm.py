"""LLM catalog endpoints — powers the model-picker dropdown in the chat UI."""
from __future__ import annotations
from typing import Any

from fastapi import APIRouter

from backend.llm.registry import BEDROCK_MODELS, DEFAULT_MODEL_ID
from backend.config import settings
from backend.llm.bedrock import llm_provider_name


router = APIRouter(prefix="/api/llm", tags=["llm"])


@router.get("/models")
def list_models() -> dict[str, Any]:
    """Return the catalog of Bedrock models available + current defaults.

    Frontend uses this to render the chat-panel model dropdown. Fields per
    model match backend.llm.registry.ModelEntry.
    """
    return {
        "provider": llm_provider_name(),
        "bedrock_enabled": settings.use_bedrock,
        "anthropic_fallback_enabled": settings.has_anthropic_fallback,
        "default_model_id": DEFAULT_MODEL_ID,
        "models": BEDROCK_MODELS,
    }
