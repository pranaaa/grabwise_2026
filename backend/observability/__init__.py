"""Observability — LLM call logging, hallucination tracking, CloudWatch emission.

Designed to be no-op-safe: if AWS isn't configured, all CloudWatch emissions
silently fall back to local logger. The DB path always runs.
"""
from backend.observability.tracker import record_llm_call, flag_hallucination
from backend.observability.pricing import estimate_cost_usd

__all__ = ["record_llm_call", "flag_hallucination", "estimate_cost_usd"]
