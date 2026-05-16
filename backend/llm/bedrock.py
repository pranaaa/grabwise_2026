"""LLM factory with Bedrock-primary, Anthropic-direct as failsafe.

Pattern
-------
- If BOTH Bedrock and Anthropic credentials are configured: every invoke
  hits Bedrock first; on any exception (auth, throttle, validation, network)
  the call retries against Anthropic direct. A warning is logged so we can
  see when fallback fires.
- If only Bedrock is configured: every call goes to Bedrock; failures bubble.
- If only Anthropic is configured: every call goes to Anthropic direct.

The returned object supports the subset of LangChain's BaseChatModel API our
agents actually use:
  - `.bind_tools(tools, **kwargs)`        — used by agents
  - `.with_structured_output(schema, **kw)` — used by supervisor
  - `.invoke(messages, **kwargs)`         — used everywhere
  - `.ainvoke(messages, **kwargs)`        — async; for future
These are forwarded to BOTH the primary and the fallback so tool/schema binding
is preserved end-to-end.
"""
from __future__ import annotations
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from backend.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider builders
# ---------------------------------------------------------------------------
def _bedrock(model_id: str, **kwargs: Any) -> BaseChatModel:
    """Construct a Bedrock chat model.

    We pass credentials explicitly so the call doesn't depend on the default
    boto3 lookup chain — easier to debug and won't accidentally pick up a
    different profile.
    """
    from langchain_aws import ChatBedrockConverse

    bedrock_kwargs: dict[str, Any] = dict(
        model=model_id,
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )
    # Only pass session token if present — long-term IAM users don't have one.
    if settings.aws_session_token:
        bedrock_kwargs["aws_session_token"] = settings.aws_session_token
    bedrock_kwargs.update(kwargs)
    return ChatBedrockConverse(**bedrock_kwargs)


def _anthropic(model: str, **kwargs: Any) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "Anthropic fallback unavailable — set ANTHROPIC_API_KEY in .env"
        )
    return ChatAnthropic(
        model=model,
        api_key=settings.anthropic_api_key,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Fallback proxy
# ---------------------------------------------------------------------------
class FallbackChatModel:
    """Bedrock-first, Anthropic-fallback chat-model proxy.

    Mirrors just enough of LangChain's BaseChatModel surface for our agents
    and supervisor. Each binding method (bind_tools / with_structured_output)
    is applied to BOTH the primary and the fallback so that, regardless of
    which one ends up actually answering, the same tools / output schema are
    available.
    """

    def __init__(self, primary: Runnable, fallback: Runnable | None):
        self.primary = primary
        self.fallback = fallback

    # ---- Binding helpers — propagate to BOTH primary and fallback ----
    def bind_tools(self, tools: list, **kwargs: Any) -> "FallbackChatModel":
        return FallbackChatModel(
            primary=self.primary.bind_tools(tools, **kwargs),
            fallback=self.fallback.bind_tools(tools, **kwargs) if self.fallback is not None else None,
        )

    def with_structured_output(self, schema: Any, **kwargs: Any) -> "FallbackChatModel":
        return FallbackChatModel(
            primary=self.primary.with_structured_output(schema, **kwargs),
            fallback=self.fallback.with_structured_output(schema, **kwargs) if self.fallback is not None else None,
        )

    def bind(self, **kwargs: Any) -> "FallbackChatModel":
        return FallbackChatModel(
            primary=self.primary.bind(**kwargs),
            fallback=self.fallback.bind(**kwargs) if self.fallback is not None else None,
        )

    # ---- Invocation — try primary, retry on fallback if it errors ----
    def invoke(self, *args: Any, **kwargs: Any):
        try:
            return self.primary.invoke(*args, **kwargs)
        except Exception as e:
            if self.fallback is None:
                raise
            logger.warning(
                "Bedrock primary failed (%s: %s) — retrying on Anthropic direct fallback",
                type(e).__name__, str(e)[:240],
            )
            return self.fallback.invoke(*args, **kwargs)

    async def ainvoke(self, *args: Any, **kwargs: Any):
        try:
            return await self.primary.ainvoke(*args, **kwargs)
        except Exception as e:
            if self.fallback is None:
                raise
            logger.warning(
                "Bedrock primary failed (%s: %s) — retrying on Anthropic direct fallback",
                type(e).__name__, str(e)[:240],
            )
            return await self.fallback.ainvoke(*args, **kwargs)

    # ---- Streaming helpers — best-effort: stream from whichever responds ----
    def stream(self, *args: Any, **kwargs: Any):
        try:
            yield from self.primary.stream(*args, **kwargs)
            return
        except Exception as e:
            if self.fallback is None:
                raise
            logger.warning(
                "Bedrock primary stream failed (%s: %s) — falling back to Anthropic direct",
                type(e).__name__, str(e)[:240],
            )
            yield from self.fallback.stream(*args, **kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_sonnet(model_override: str | None = None, **kwargs: Any) -> Any:
    """Sonnet-tier model — agent reasoning. Bedrock primary, Anthropic fallback.

    Args:
        model_override: Optional Bedrock model_id to use instead of the
            default `BEDROCK_MODEL_SONNET`. Per-request UI selection threads
            through here. Anthropic fallback always uses its configured model.
    """
    bedrock_ok = settings.use_bedrock
    anthropic_ok = settings.has_anthropic_fallback
    bedrock_id = model_override or settings.bedrock_model_sonnet

    if bedrock_ok and anthropic_ok:
        return FallbackChatModel(
            primary=_bedrock(bedrock_id, **kwargs),
            fallback=_anthropic(settings.anthropic_model_sonnet, **kwargs),
        )
    if bedrock_ok:
        return _bedrock(bedrock_id, **kwargs)
    if anthropic_ok:
        return _anthropic(settings.anthropic_model_sonnet, **kwargs)
    raise RuntimeError(
        "No LLM credentials found. Configure AWS Bedrock (AWS_ACCESS_KEY_ID + "
        "AWS_SECRET_ACCESS_KEY + AWS_REGION, plus AWS_SESSION_TOKEN if STS) "
        "and/or ANTHROPIC_API_KEY in .env"
    )


def get_haiku(model_override: str | None = None, **kwargs: Any) -> Any:
    """Haiku-tier model — supervisor routing. Bedrock primary, Anthropic fallback."""
    bedrock_ok = settings.use_bedrock
    anthropic_ok = settings.has_anthropic_fallback
    bedrock_id = model_override or settings.bedrock_model_haiku

    if bedrock_ok and anthropic_ok:
        return FallbackChatModel(
            primary=_bedrock(bedrock_id, **kwargs),
            fallback=_anthropic(settings.anthropic_model_haiku, **kwargs),
        )
    if bedrock_ok:
        return _bedrock(bedrock_id, **kwargs)
    if anthropic_ok:
        return _anthropic(settings.anthropic_model_haiku, **kwargs)
    raise RuntimeError("No LLM credentials found — see get_sonnet() error message")


def llm_provider_name() -> str:
    """Human-readable provider label, surfaced in /health and the UI badge."""
    bedrock = settings.use_bedrock
    anthropic = settings.has_anthropic_fallback
    if bedrock and anthropic:
        return "bedrock+anthropic-fallback"
    if bedrock:
        return "bedrock"
    if anthropic:
        return "anthropic-direct"
    return "no-provider"
