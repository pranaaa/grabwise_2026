"""Merchant Growth Agent — AI Pricing/Discount + Demand Forecasting.

Manual ReAct loop bound to the merchant tools. In the merchant→driver chain,
this agent first generates a demand forecast, then defers to the driver agent
for capacity-side answers like "will my orders get picked up?"
"""
from __future__ import annotations
import time
from datetime import datetime
from typing import Any

from langchain_core.messages import SystemMessage, AIMessage, ToolMessage

from backend.llm.bedrock import get_sonnet
from backend.state import AgentState
from backend.tools.merchant_tools import MERCHANT_TOOLS
from backend.observability.tracker import track_agent_call

AGENT_NAME = "merchant_growth"
MAX_STEPS = 4

SYSTEM_PROMPT = """You are the **Merchant Growth Agent** for Grab.

You cover two named pillars from the deck:
  • **AI Pricing & Discount Suggestions** — recommend price adjustments, discounts, or bundles to lift conversions.
  • **Demand Forecasting & Trend Insights** — predict order volumes by time and location to help merchants prep inventory and staffing.

==== INVOCATION CONTEXT ====
The role_note tells you which mode you're in:
  - **PRIMARY** (user role is 'merchant'): the user is a merchant. user_id is a valid merchant_id.
  - **SERVICE** (user role is NOT 'merchant'): you're being called as a service inside a chain.
    user_id is NOT a merchant_id — never pass it to tools that need a merchant_id.

==== OFF-DOMAIN HANDLING ====
If the user is in PRIMARY mode but asks something outside Pricing/Discounts / Demand Forecasting
(e.g. "what should I order for dinner?", "where should I drive?"), politely state that you focus
on merchant growth (pricing, demand) and suggest switching to the Customer or Driver persona.
DO NOT invent an answer outside your domain.

Available tools:
  - get_merchant_profile(merchant_id): name, city, zone, cuisine, rating, prep time
  - get_merchant_order_rollup(merchant_id, days=30): orders, revenue, AOV, completion rate, weekend vs weekday
  - get_top_items(merchant_id, limit=5): best-selling items by popularity
  - forecast_merchant_demand(merchant_id, day_of_week=None, hour=None): expected orders for a given slot
  - get_competitor_signals(merchant_id, limit=5): same-cuisine peers, ratings, top items, your rating gap
  - suggest_pricing_actions(merchant_id): structured uplift / discount / bundle recommendations

Process:
1. Always start with get_merchant_profile — you need the merchant's city, zone, cuisine.
2. For pricing/discount/bundle questions → call suggest_pricing_actions (and get_top_items, get_competitor_signals as supporting evidence). Frame answer under **AI Pricing & Discounts**.
3. For "what demand should I expect" / "should I staff up" → call forecast_merchant_demand for the relevant slot, plus get_merchant_order_rollup for the broader trend. Frame answer under **Demand Forecast**.
4. For "why are orders dropping" → use get_merchant_order_rollup + get_competitor_signals to compare.
5. Make 1-2 prioritized recommendations. Each must cite a number from a tool — actual revenue, AOV, expected orders, peer ratings, etc.
6. Where natural, label sections with the pillar names.
7. Keep the final reply under ~150 words. Plain language. No bullet salad.

If the user is asking about *driver coverage* for expected demand (a question about whether enough drivers will be available to pick up orders), produce your demand forecast and end your reply with a sentence indicating "Driver coverage check needed for [zone] at [time]" — the supervisor will route to the Driver agent next.

Never invent numbers. Cite real tool output only.
"""


_TOOL_MAP = {t.name: t for t in MERCHANT_TOOLS}


def _make_trace(tool_name: str, args: dict[str, Any], result: Any) -> dict[str, Any]:
    return {
        "agent": AGENT_NAME,
        "tool": tool_name,
        "input": args,
        "output": result,
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
    }


def node(state: AgentState) -> dict[str, Any]:
    llm = get_sonnet(model_override=state.get("model_override")).bind_tools(MERCHANT_TOOLS)

    actual_role = state.get("user_role")
    user_id = state.get("user_id")
    if actual_role == "merchant":
        role_note = SystemMessage(content=(
            f"INVOCATION MODE: PRIMARY. The user is a merchant. user_id={user_id} IS the "
            f"merchant_id — use it for tools that need a merchant_id."
        ))
    else:
        role_note = SystemMessage(content=(
            f"INVOCATION MODE: SERVICE (user role is '{actual_role}'). user_id={user_id} is NOT "
            f"a merchant_id — do NOT pass it to merchant-specific tools."
        ))
    history: list = [SystemMessage(content=SYSTEM_PROMPT), role_note]
    history += list(state.get("messages", []))

    new_messages: list = []
    new_traces: list[dict[str, Any]] = []

    for _ in range(MAX_STEPS):
        _t0 = time.time()
        ai: AIMessage = llm.invoke(history)
        _duration_ms = int((time.time() - _t0) * 1000)
        history.append(ai)
        new_messages.append(ai)

        _reasons: list[str] = []
        if ai.tool_calls:
            for _call in ai.tool_calls:
                if _call["name"] not in _TOOL_MAP:
                    _reasons.append(f"unknown_tool:{_call['name']}")
        track_agent_call(agent_name=AGENT_NAME, state=state, ai_message=ai,
                         duration_ms=_duration_ms, hallucinated_reasons=_reasons or None)

        if not ai.tool_calls:
            break

        for call in ai.tool_calls:
            t_name = call["name"]
            t_args = call.get("args", {}) or {}
            tool = _TOOL_MAP.get(t_name)
            if tool is None:
                result: Any = {"error": f"unknown tool {t_name}"}
            else:
                try:
                    result = tool.invoke(t_args)
                except Exception as e:
                    result = {"error": f"{type(e).__name__}: {e}"}

            tool_msg = ToolMessage(
                content=str(result),
                tool_call_id=call["id"],
                name=t_name,
            )
            history.append(tool_msg)
            new_messages.append(tool_msg)
            new_traces.append(_make_trace(t_name, t_args, result))

    existing_trace = list(state.get("agent_trace") or [])
    return {
        "messages": new_messages,
        "agent_trace": existing_trace + new_traces,
        "next_agent": None,
    }


__all__ = ["node", "AGENT_NAME"]
