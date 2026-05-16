"""CloudWatch emission — best-effort metric + log writes.

Two channels:
  1. **CloudWatch Metrics** (namespace `GrabWise/LLM`)
     - `LLMCalls` (Count) by `ModelId`, `Hallucinated`, `Agent`
     - `LLMTokensInput`, `LLMTokensOutput` (Count) by `ModelId`
     - `LLMCostUSD` (None) by `ModelId`
     - `LLMHallucination` (Count) by `ModelId`, `ReasonCode`
  2. **CloudWatch Logs** (log group `/grabwise/llm-hallucinations`)
     - Structured JSON events for hallucinations (forensics)

If AWS credentials aren't configured (or boto3 isn't importable), every
function in this module no-ops gracefully — the agent path is never blocked
by CloudWatch errors. We log to the local Python logger instead.
"""
from __future__ import annotations
import json
import logging
import time
from typing import Any

from backend.config import settings

logger = logging.getLogger(__name__)

NAMESPACE = "GrabWise/LLM"
LOG_GROUP = "/grabwise/llm-hallucinations"
LOG_STREAM_PREFIX = "agent"

# --- Lazy boto3 clients ----------------------------------------------------
_cw_client = None
_cw_logs_client = None
_log_stream_initialized = False


def _aws_creds_present() -> bool:
    return bool(settings.aws_access_key_id and settings.aws_secret_access_key and settings.aws_region)


def _get_cw():
    """Get-or-create CloudWatch Metrics client. None if AWS not configured."""
    global _cw_client
    if _cw_client is not None:
        return _cw_client
    if not _aws_creds_present():
        return None
    try:
        import boto3
        kwargs = {
            "region_name": settings.aws_region,
            "aws_access_key_id": settings.aws_access_key_id,
            "aws_secret_access_key": settings.aws_secret_access_key,
        }
        if settings.aws_session_token:
            kwargs["aws_session_token"] = settings.aws_session_token
        _cw_client = boto3.client("cloudwatch", **kwargs)
        return _cw_client
    except Exception as e:
        logger.warning("CloudWatch metrics client init failed: %s", e)
        return None


def _get_cw_logs():
    """Get-or-create CloudWatch Logs client."""
    global _cw_logs_client
    if _cw_logs_client is not None:
        return _cw_logs_client
    if not _aws_creds_present():
        return None
    try:
        import boto3
        kwargs = {
            "region_name": settings.aws_region,
            "aws_access_key_id": settings.aws_access_key_id,
            "aws_secret_access_key": settings.aws_secret_access_key,
        }
        if settings.aws_session_token:
            kwargs["aws_session_token"] = settings.aws_session_token
        _cw_logs_client = boto3.client("logs", **kwargs)
        return _cw_logs_client
    except Exception as e:
        logger.warning("CloudWatch logs client init failed: %s", e)
        return None


def _ensure_log_group_and_stream() -> str | None:
    """Best-effort ensure the log group + a per-process stream exist. Returns stream name."""
    global _log_stream_initialized
    cw = _get_cw_logs()
    if cw is None:
        return None
    stream_name = f"{LOG_STREAM_PREFIX}-{int(time.time())}"
    if _log_stream_initialized:
        return stream_name
    try:
        try:
            cw.create_log_group(logGroupName=LOG_GROUP)
        except Exception:
            pass  # already exists
        try:
            cw.create_log_stream(logGroupName=LOG_GROUP, logStreamName=stream_name)
        except Exception:
            pass
        _log_stream_initialized = True
        return stream_name
    except Exception as e:
        logger.warning("CloudWatch log group/stream init failed: %s", e)
        return None


# --- Public emission API ---------------------------------------------------
def emit_call_metrics(*, model_id: str, agent: str | None, hallucinated: bool,
                     input_tokens: int | None, output_tokens: int | None,
                     cost_usd: float | None) -> None:
    """Emit per-call CloudWatch metrics. No-op if AWS not configured."""
    cw = _get_cw()
    if cw is None:
        return
    dims = [{"Name": "ModelId", "Value": model_id}]
    if agent:
        dims_with_agent = dims + [{"Name": "Agent", "Value": agent}]
    else:
        dims_with_agent = dims

    metric_data: list[dict[str, Any]] = [
        {"MetricName": "LLMCalls", "Dimensions": dims_with_agent, "Value": 1, "Unit": "Count"},
    ]
    if input_tokens is not None:
        metric_data.append({"MetricName": "LLMTokensInput", "Dimensions": dims, "Value": int(input_tokens), "Unit": "Count"})
    if output_tokens is not None:
        metric_data.append({"MetricName": "LLMTokensOutput", "Dimensions": dims, "Value": int(output_tokens), "Unit": "Count"})
    if cost_usd is not None:
        metric_data.append({"MetricName": "LLMCostUSD", "Dimensions": dims, "Value": float(cost_usd), "Unit": "None"})
    if hallucinated:
        metric_data.append({"MetricName": "LLMHallucination", "Dimensions": dims, "Value": 1, "Unit": "Count"})

    try:
        cw.put_metric_data(Namespace=NAMESPACE, MetricData=metric_data)
    except Exception as e:
        logger.warning("CloudWatch put_metric_data failed: %s", e)


def emit_hallucination_log(*, model_id: str, agent: str | None,
                            reasons: list[str], context: dict[str, Any] | None = None) -> None:
    """Emit a structured hallucination event to CloudWatch Logs."""
    payload = {
        "model_id": model_id,
        "agent": agent,
        "reasons": reasons,
        "context": context or {},
        "ts": int(time.time() * 1000),
    }
    logger.warning("LLM hallucination: %s", json.dumps(payload))

    cw = _get_cw_logs()
    if cw is None:
        return
    stream = _ensure_log_group_and_stream()
    if not stream:
        return
    try:
        cw.put_log_events(
            logGroupName=LOG_GROUP,
            logStreamName=stream,
            logEvents=[{"timestamp": payload["ts"], "message": json.dumps(payload)}],
        )
    except Exception as e:
        logger.warning("CloudWatch put_log_events failed: %s", e)
