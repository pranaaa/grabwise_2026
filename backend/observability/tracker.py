"""LLM call tracker — writes one row per invocation, emits CloudWatch.

Public surface:
  - record_llm_call(...) — call this once per LLM invocation (from agents)
  - flag_hallucination(...) — convenience for "we already wrote a call, but
    later detected something hallucinated; update the row + emit a metric"

Both are best-effort: any DB / AWS failure is swallowed so the agent path
never breaks because telemetry failed.
"""
from __future__ import annotations
import logging
from typing import Any

from backend.db.database import get_session
from backend.db import models as M
from backend.observability.pricing import estimate_cost_usd
from backend.observability.cloudwatch import emit_call_metrics, emit_hallucination_log

logger = logging.getLogger(__name__)


def record_llm_call(
    *,
    model_id: str,
    provider: str,
    agent: str | None = None,
    auth_user_id: int | None = None,
    duration_ms: int | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    hallucinated: bool = False,
    reasons: list[str] | None = None,
    error: str | None = None,
) -> int | None:
    """Persist one row to llm_call_logs and emit CloudWatch metrics.

    Returns the created row id, or None on failure (always safe).
    """
    try:
        cost = estimate_cost_usd(model_id, input_tokens, output_tokens)
        with get_session() as s:
            row = M.LLMCallLog(
                model_id=model_id,
                provider=provider,
                agent=agent,
                auth_user_id=auth_user_id,
                duration_ms=duration_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=cost,
                hallucinated=hallucinated,
                reasons_json=reasons or None,
                error=(error[:255] if error else None),
            )
            s.add(row)
            s.flush()
            row_id = row.id

        # Best-effort CloudWatch
        try:
            emit_call_metrics(
                model_id=model_id,
                agent=agent,
                hallucinated=hallucinated,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
            )
            if hallucinated and reasons:
                emit_hallucination_log(model_id=model_id, agent=agent, reasons=reasons,
                                       context={"row_id": row_id, "duration_ms": duration_ms})
        except Exception as e:
            logger.warning("CloudWatch emission failed (non-fatal): %s", e)

        return row_id
    except Exception as e:
        logger.warning("record_llm_call failed (non-fatal): %s", e)
        return None


def _extract_tokens(ai_message: Any) -> tuple[int | None, int | None]:
    """Pull (input_tokens, output_tokens) from a LangChain AIMessage.

    Different providers stash usage in different places — we try common shapes.
    """
    if ai_message is None:
        return (None, None)
    usage = getattr(ai_message, "usage_metadata", None)
    if isinstance(usage, dict):
        return (usage.get("input_tokens"), usage.get("output_tokens"))
    if usage is not None:
        return (getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None))
    # Some providers stash under response_metadata.usage.{input,output}_tokens
    meta = getattr(ai_message, "response_metadata", {}) or {}
    usage = meta.get("usage") if isinstance(meta, dict) else None
    if isinstance(usage, dict):
        return (usage.get("input_tokens") or usage.get("prompt_tokens"),
                usage.get("output_tokens") or usage.get("completion_tokens"))
    return (None, None)


def track_agent_call(
    *,
    agent_name: str,
    state: dict | None,
    ai_message: Any = None,
    duration_ms: int | None = None,
    hallucinated_reasons: list[str] | None = None,
) -> None:
    """Convenience for an agent ReAct-loop LLM call.

    Pulls model_id from state's `model_override` (else the .env default),
    extracts token metadata from the AIMessage, and persists + emits.
    """
    from backend.config import settings  # local import to avoid cycle on cold start
    model_id = (state or {}).get("model_override") or settings.bedrock_model_sonnet or "unknown"
    provider = "bedrock" if not (model_id.startswith("claude-") or model_id.startswith("anthropic.")) else "bedrock"
    if model_id.startswith("claude-"):
        provider = "anthropic-direct"
    in_tok, out_tok = _extract_tokens(ai_message)
    record_llm_call(
        model_id=model_id,
        provider=provider,
        agent=agent_name,
        duration_ms=duration_ms,
        input_tokens=in_tok,
        output_tokens=out_tok,
        hallucinated=bool(hallucinated_reasons),
        reasons=hallucinated_reasons or None,
    )


def flag_hallucination(
    *,
    model_id: str,
    agent: str | None,
    reasons: list[str],
    context: dict[str, Any] | None = None,
) -> None:
    """Lightweight standalone hallucination event — for places that don't
    track full call metadata (e.g. supervisor schema slips).

    Writes a minimal row marking hallucinated=True and emits the CW log.
    """
    try:
        with get_session() as s:
            s.add(M.LLMCallLog(
                model_id=model_id,
                provider="bedrock",          # caller can override via record_llm_call if needed
                agent=agent,
                hallucinated=True,
                reasons_json=reasons,
            ))
    except Exception as e:
        logger.warning("flag_hallucination DB write failed: %s", e)
    try:
        emit_hallucination_log(model_id=model_id, agent=agent, reasons=reasons, context=context)
    except Exception as e:
        logger.warning("flag_hallucination CloudWatch emit failed: %s", e)
