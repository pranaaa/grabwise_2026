"""Bedrock model registry — catalog of models available to GrabWise.

Only includes models the workshop account has access to. Embedding-only and
vision-only models are excluded (our agents are text + tool calling).

Each entry carries metadata the frontend uses to render the dropdown:
  - id:           Bedrock model ID (passed to ChatBedrockConverse(model=...))
  - name:         human-readable label
  - category:     "Reasoning" | "Fast" | "Code" — group in the dropdown
  - size:         "Small" | "Medium" | "Large" — hint at speed/quality tradeoff
  - description:  one-liner for the option tooltip
  - tools_likely: best-guess whether tool calling works via Bedrock's Converse API
  - recommended:  shown with a star in the dropdown; safe default

Tool-calling availability on Bedrock varies by model + region — these flags are
best-effort. The Anthropic-direct fallback handles errors transparently when a
chosen Bedrock model rejects tool calls.
"""
from __future__ import annotations
from typing import TypedDict


class ModelEntry(TypedDict):
    id: str
    name: str
    category: str          # "Reasoning" | "Fast" | "Code"
    size: str              # "Small" | "Medium" | "Large"
    description: str
    tools_likely: bool
    recommended: bool


BEDROCK_MODELS: list[ModelEntry] = [
    # =========================================================================
    # Reasoning — best fit for agent reasoning + tool calling
    # =========================================================================
    {
        "id": "qwen.qwen3-235b-a22b-2507-v1:0",
        "name": "Qwen3 235B (2507)",
        "category": "Reasoning",
        "size": "Large",
        "description": "Alibaba flagship — strongest reasoning + native tool calling.",
        "tools_likely": True,
        "recommended": True,
    },
    {
        "id": "deepseek.v3.2",
        "name": "DeepSeek V3.2",
        "category": "Reasoning",
        "size": "Large",
        "description": "DeepSeek's latest — strong on reasoning + math.",
        "tools_likely": True,
        "recommended": True,
    },
    {
        "id": "qwen.qwen3-next-80b-a3b",
        "name": "Qwen3 Next 80B",
        "category": "Reasoning",
        "size": "Large",
        "description": "Newer Qwen3 architecture, mid-tier size.",
        "tools_likely": True,
        "recommended": False,
    },
    {
        "id": "zai.glm-5",
        "name": "GLM 5",
        "category": "Reasoning",
        "size": "Large",
        "description": "Zhipu AI flagship — multilingual + tool calling.",
        "tools_likely": True,
        "recommended": False,
    },
    {
        "id": "qwen.qwen3-32b-v1:0",
        "name": "Qwen3 32B",
        "category": "Reasoning",
        "size": "Medium",
        "description": "Faster Qwen3 variant — good speed/quality balance.",
        "tools_likely": True,
        "recommended": True,
    },
    {
        "id": "deepseek.v3-v1:0",
        "name": "DeepSeek V3.1",
        "category": "Reasoning",
        "size": "Large",
        "description": "Previous DeepSeek V3 generation.",
        "tools_likely": True,
        "recommended": False,
    },
    {
        "id": "nvidia.nemotron-super-3-120b",
        "name": "Nemotron Super 120B",
        "category": "Reasoning",
        "size": "Large",
        "description": "NVIDIA's flagship reasoning model.",
        "tools_likely": True,
        "recommended": False,
    },

    # =========================================================================
    # Fast — smaller models, lower latency, may have weaker tool calling
    # =========================================================================
    {
        "id": "moonshotai.kimi-k2.5",
        "name": "Kimi K2.5",
        "category": "Fast",
        "size": "Medium",
        "description": "Moonshot AI — long-context, mid-size.",
        "tools_likely": True,
        "recommended": False,
    },
    {
        "id": "zai.glm-4.7",
        "name": "GLM 4.7",
        "category": "Fast",
        "size": "Medium",
        "description": "Previous GLM generation — proven tool calling.",
        "tools_likely": True,
        "recommended": False,
    },
    {
        "id": "zai.glm-4.7-flash",
        "name": "GLM 4.7 Flash",
        "category": "Fast",
        "size": "Small",
        "description": "Fast inference variant — supervisor-class latency.",
        "tools_likely": True,
        "recommended": False,
    },
    {
        "id": "google.gemma-3-27b-it",
        "name": "Gemma 3 27B",
        "category": "Fast",
        "size": "Medium",
        "description": "Google's open-weight model — limited tool support.",
        "tools_likely": False,
        "recommended": False,
    },
    {
        "id": "google.gemma-3-12b-it",
        "name": "Gemma 3 12B",
        "category": "Fast",
        "size": "Small",
        "description": "Smaller Gemma 3 — fast, limited tool support.",
        "tools_likely": False,
        "recommended": False,
    },
    {
        "id": "google.gemma-3-4b-it",
        "name": "Gemma 3 4B",
        "category": "Fast",
        "size": "Small",
        "description": "Smallest Gemma 3 — quick replies, no tool calling.",
        "tools_likely": False,
        "recommended": False,
    },
    {
        "id": "nvidia.nemotron-nano-3-30b",
        "name": "Nemotron Nano 30B",
        "category": "Fast",
        "size": "Medium",
        "description": "NVIDIA's lighter Nemotron variant.",
        "tools_likely": True,
        "recommended": False,
    },
    {
        "id": "nvidia.nemotron-nano-12b-v2",
        "name": "Nemotron Nano 12B",
        "category": "Fast",
        "size": "Small",
        "description": "Compact NVIDIA model.",
        "tools_likely": False,
        "recommended": False,
    },
    {
        "id": "nvidia.nemotron-nano-9b-v2",
        "name": "Nemotron Nano 9B",
        "category": "Fast",
        "size": "Small",
        "description": "Smallest NVIDIA Nemotron — fast, limited.",
        "tools_likely": False,
        "recommended": False,
    },

    # =========================================================================
    # Code — specialized models, included for completeness
    # =========================================================================
    {
        "id": "qwen.qwen3-coder-480b-a35b-v1:0",
        "name": "Qwen3 Coder 480B",
        "category": "Code",
        "size": "Large",
        "description": "Largest code-specialized Qwen3.",
        "tools_likely": True,
        "recommended": False,
    },
    {
        "id": "qwen.qwen3-coder-30b-a3b-v1:0",
        "name": "Qwen3 Coder 30B",
        "category": "Code",
        "size": "Medium",
        "description": "Compact code-focused model.",
        "tools_likely": True,
        "recommended": False,
    },
]


# Default model used when the request doesn't specify one. Picked to be a
# reasonable balance of quality + speed + tool-calling reliability.
DEFAULT_MODEL_ID = "qwen.qwen3-32b-v1:0"


def find(model_id: str) -> ModelEntry | None:
    for m in BEDROCK_MODELS:
        if m["id"] == model_id:
            return m
    return None


def is_valid(model_id: str | None) -> bool:
    if not model_id:
        return False
    return find(model_id) is not None
