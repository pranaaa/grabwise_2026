"""Bedrock model pricing — best-effort cost estimation.

Numbers are approximate AWS Bedrock on-demand pricing in us-west-2 as of
mid-2026 in USD per 1,000 tokens. Final billing comes from your AWS account;
this is for *relative* cost comparisons in the admin dashboard. Update as
AWS publishes new rates.

If a model isn't in this table we fall back to MODEL_COST_FALLBACK so the
admin widget always shows *some* cost estimate rather than a blank.
"""
from __future__ import annotations
from typing import TypedDict


class TokenPrice(TypedDict):
    input_per_1k: float    # USD per 1,000 input tokens
    output_per_1k: float   # USD per 1,000 output tokens


# Per-model pricing. Source: AWS Bedrock console / pricing page (best effort).
MODEL_PRICES: dict[str, TokenPrice] = {
    # --- Qwen3 family ---
    "qwen.qwen3-235b-a22b-2507-v1:0":   {"input_per_1k": 0.0011,  "output_per_1k": 0.0033},
    "qwen.qwen3-next-80b-a3b":          {"input_per_1k": 0.0006,  "output_per_1k": 0.0018},
    "qwen.qwen3-32b-v1:0":              {"input_per_1k": 0.00015, "output_per_1k": 0.0006},
    "qwen.qwen3-coder-480b-a35b-v1:0":  {"input_per_1k": 0.0011,  "output_per_1k": 0.0033},
    "qwen.qwen3-coder-30b-a3b-v1:0":    {"input_per_1k": 0.00015, "output_per_1k": 0.0006},
    # --- DeepSeek ---
    "deepseek.v3.2":      {"input_per_1k": 0.00027, "output_per_1k": 0.0011},
    "deepseek.v3-v1:0":   {"input_per_1k": 0.00027, "output_per_1k": 0.0011},
    # --- Zhipu GLM ---
    "zai.glm-5":          {"input_per_1k": 0.0006, "output_per_1k": 0.0018},
    "zai.glm-4.7":        {"input_per_1k": 0.0003, "output_per_1k": 0.0009},
    "zai.glm-4.7-flash":  {"input_per_1k": 0.00010, "output_per_1k": 0.0003},
    # --- Moonshot ---
    "moonshotai.kimi-k2.5": {"input_per_1k": 0.0005, "output_per_1k": 0.0015},
    # --- Google Gemma ---
    "google.gemma-3-27b-it": {"input_per_1k": 0.00025, "output_per_1k": 0.0008},
    "google.gemma-3-12b-it": {"input_per_1k": 0.00015, "output_per_1k": 0.0005},
    "google.gemma-3-4b-it":  {"input_per_1k": 0.00006, "output_per_1k": 0.00018},
    # --- NVIDIA Nemotron ---
    "nvidia.nemotron-super-3-120b": {"input_per_1k": 0.0008, "output_per_1k": 0.0024},
    "nvidia.nemotron-nano-3-30b":   {"input_per_1k": 0.00018, "output_per_1k": 0.0006},
    "nvidia.nemotron-nano-12b-v2":  {"input_per_1k": 0.00012, "output_per_1k": 0.00036},
    "nvidia.nemotron-nano-9b-v2":   {"input_per_1k": 0.00009, "output_per_1k": 0.00027},
    # --- Anthropic direct fallback (rough Anthropic API pricing) ---
    "claude-sonnet-4-5":  {"input_per_1k": 0.003,  "output_per_1k": 0.015},
    "claude-haiku-4-5":   {"input_per_1k": 0.0008, "output_per_1k": 0.004},
}

# Fallback for models we don't have explicit prices for
MODEL_COST_FALLBACK: TokenPrice = {"input_per_1k": 0.0005, "output_per_1k": 0.0015}


def get_price(model_id: str) -> TokenPrice:
    return MODEL_PRICES.get(model_id, MODEL_COST_FALLBACK)


def estimate_cost_usd(model_id: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    """Compute cost in USD for one LLM call. Returns None if no token data."""
    if input_tokens is None and output_tokens is None:
        return None
    p = get_price(model_id)
    cost = 0.0
    cost += (input_tokens or 0) / 1000.0 * p["input_per_1k"]
    cost += (output_tokens or 0) / 1000.0 * p["output_per_1k"]
    return round(cost, 6)
