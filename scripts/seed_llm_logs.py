"""Populate `llm_call_logs` with realistic placeholder data so the admin
LLM Performance & Cost widget looks populated before any real chat usage.

Usage
-----
    # Add ~1000 rows on top of whatever's already there
    python -m scripts.seed_llm_logs

    # Wipe existing rows first
    python -m scripts.seed_llm_logs --clear

    # Custom volume
    python -m scripts.seed_llm_logs --count 2500

Distribution
------------
Each row is randomly assigned a model, agent, timestamp (skewed recent),
token counts, latency, and a per-model hallucination probability tuned to
match each model's reputation:

  ★ Reasoning (Qwen3 235B, DeepSeek V3.2, Qwen3 32B)  → 3-8%   halluc rate
  Other reasoning (Qwen3 Next 80B, GLM 5, etc.)        → 8-15%
  Fast / smaller (Kimi K2.5, GLM 4.7 Flash, Nemotron)  → 15-25%
  Weak (Gemma 3 small, Nemotron Nano 9B)               → 35-55%
  Code-specialized (Qwen3 Coder 480B/30B)              → 30-45%

Cost is computed via backend.observability.pricing so the widget shows
internally-consistent numbers.

Idempotency
-----------
Safe to run multiple times. Use --clear to start from a clean slate.
Real LLM calls accumulate alongside the placeholder rows.
"""
from __future__ import annotations
import argparse
import random
from datetime import datetime, timedelta

from backend.db.database import get_session, init_db
from backend.db import models as M
from backend.observability.pricing import estimate_cost_usd


# ---------------------------------------------------------------------------
# Per-model demo profile: (model_id, weight, hall_rate, avg_in, avg_out, avg_lat_ms)
# weight     → relative call volume share
# hall_rate  → probability this call is flagged hallucinated
# avg_in/out → token means (log-normal-ish noise applied below)
# avg_lat_ms → latency mean
# ---------------------------------------------------------------------------
MODEL_PROFILES: list[tuple[str, float, float, int, int, int]] = [
    # ★ Recommended reasoning — high volume, low hallucination
    ("qwen.qwen3-235b-a22b-2507-v1:0", 0.18, 0.04, 1400,  600, 2200),
    ("qwen.qwen3-32b-v1:0",            0.22, 0.07, 1100,  480, 1400),
    ("deepseek.v3.2",                  0.14, 0.05, 1300,  520, 1900),
    # Other reasoning — medium volume, medium hallucination
    ("qwen.qwen3-next-80b-a3b",        0.06, 0.10, 1200,  500, 1700),
    ("zai.glm-5",                      0.05, 0.11, 1150,  520, 1800),
    ("deepseek.v3-v1:0",               0.04, 0.10, 1200,  500, 1700),
    ("nvidia.nemotron-super-3-120b",   0.04, 0.13, 1300,  480, 2100),
    # Fast / smaller — lower volume, higher hallucination
    ("moonshotai.kimi-k2.5",           0.04, 0.18, 950,   400, 950),
    ("zai.glm-4.7",                    0.03, 0.20, 900,   380, 1000),
    ("zai.glm-4.7-flash",              0.03, 0.24, 800,   340, 600),
    ("nvidia.nemotron-nano-3-30b",     0.03, 0.22, 900,   360, 900),
    # Weak — tried, mostly failed
    ("google.gemma-3-27b-it",          0.02, 0.42, 850,   350, 1200),
    ("google.gemma-3-12b-it",          0.02, 0.48, 800,   320, 800),
    ("google.gemma-3-4b-it",           0.01, 0.55, 700,   280, 500),
    ("nvidia.nemotron-nano-12b-v2",    0.02, 0.38, 750,   310, 700),
    ("nvidia.nemotron-nano-9b-v2",     0.01, 0.42, 700,   280, 600),
    # Code-specialized — high hallucination on agent tasks
    ("qwen.qwen3-coder-480b-a35b-v1:0", 0.02, 0.45, 1100, 480, 2400),
    ("qwen.qwen3-coder-30b-a3b-v1:0",   0.01, 0.40, 950,  380, 1100),
]

# Which agent made the call. Supervisor is most common (every chat has one).
AGENT_WEIGHTS: list[tuple[str, float]] = [
    ("supervisor",            0.45),
    ("driver_success",        0.20),
    ("customer_convenience",  0.15),
    ("merchant_growth",       0.12),
    ("fraud_risk",            0.08),
]

# Hallucination reason codes, weighted by realism
REASON_WEIGHTS: list[tuple[str, float]] = [
    ("unknown_tool:get_city_benchmark",     0.16),
    ("unknown_tool:fetch_real_time_data",   0.12),
    ("unknown_tool:FINISH",                 0.08),
    ("unknown_tool:lookup_user_info",       0.07),
    ("invalid_agent:get_city_benchmark",    0.10),
    ("invalid_agent:demand_forecaster",     0.07),
    ("invalid_agent:supervisor",            0.05),
    ("structured_output_null",              0.18),
    ("invoke_exception:ValidationError",    0.09),
    ("invoke_exception:KeyError",           0.05),
    ("fallback_to:driver_success",          0.03),
]


def _weighted_choice(pairs):
    """Pick a value from a list of (value, weight) tuples."""
    total = sum(w for _, w in pairs)
    r = random.random() * total
    cum = 0.0
    for v, w in pairs:
        cum += w
        if r <= cum:
            return v
    return pairs[-1][0]


def _gen_reasons() -> list[str]:
    n = random.choices([1, 2, 3], weights=[0.65, 0.28, 0.07], k=1)[0]
    out: list[str] = []
    while len(out) < n:
        r = _weighted_choice(REASON_WEIGHTS)
        if r not in out:
            out.append(r)
    return out


def _provider_for(model_id: str) -> str:
    if model_id.startswith("claude-") or model_id.startswith("anthropic."):
        return "anthropic-direct" if model_id.startswith("claude-") else "bedrock"
    return "bedrock"


def seed_logs(count: int = 1000, clear: bool = False, window_hours: int = 168) -> None:
    random.seed(123)
    init_db()
    with get_session() as s:
        if clear:
            deleted = s.query(M.LLMCallLog).delete()
            s.flush()
            print(f"  ✓ cleared {deleted} existing llm_call_logs rows")

        now = datetime.utcnow()
        rows_added = 0
        for _ in range(count):
            # Pick a model by weight
            profile = _weighted_choice([(p, p[1]) for p in MODEL_PROFILES])
            model_id, _w, hall_rate, avg_in, avg_out, avg_lat = profile

            # Tokens — gaussian noise around the mean, clamped to plausible bounds
            input_tokens  = max(100, int(random.gauss(avg_in,  avg_in  * 0.25)))
            output_tokens = max(50,  int(random.gauss(avg_out, avg_out * 0.30)))

            # Latency — gaussian noise, clamped
            duration_ms = max(200, int(random.gauss(avg_lat, avg_lat * 0.25)))

            # Agent picker
            agent = _weighted_choice(AGENT_WEIGHTS)

            # Timestamp — exponential decay so most rows are recent
            hours_ago = min(window_hours, random.expovariate(1.0 / 40.0))
            invoked_at = now - timedelta(hours=hours_ago, minutes=random.randint(0, 59))

            # Hallucinated?
            hallucinated = random.random() < hall_rate
            reasons = _gen_reasons() if hallucinated else None
            error = reasons[0] if hallucinated and random.random() < 0.30 else None

            cost = estimate_cost_usd(model_id, input_tokens, output_tokens)
            provider = _provider_for(model_id)

            s.add(M.LLMCallLog(
                model_id=model_id,
                provider=provider,
                agent=agent,
                auth_user_id=None,
                invoked_at=invoked_at,
                duration_ms=duration_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=cost,
                hallucinated=hallucinated,
                reasons_json=reasons,
                error=error,
            ))
            rows_added += 1
        s.flush()
        print(f"  ✓ seeded {rows_added} llm_call_logs rows across "
              f"{len(MODEL_PROFILES)} models over the last {window_hours}h")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed placeholder LLM observability data.")
    parser.add_argument("--count",  type=int, default=1000, help="Total rows to insert (default 1000)")
    parser.add_argument("--clear",  action="store_true",   help="Wipe existing rows first")
    parser.add_argument("--window", type=int, default=168, help="Spread rows over the last N hours (default 168 = 7 days)")
    args = parser.parse_args()
    seed_logs(count=args.count, clear=args.clear, window_hours=args.window)
