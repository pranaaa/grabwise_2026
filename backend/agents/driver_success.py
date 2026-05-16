"""Driver Success Agent — earnings coach for Grab drivers.

This is a thin manual ReAct loop so we can append every tool call to
`agent_trace` (used later by the UI's Agent Activity Panel). At most
MAX_STEPS rounds, then we force a final answer.
"""
from __future__ import annotations
import time
from datetime import datetime
from typing import Any

from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage

from backend.llm.bedrock import get_sonnet
from backend.state import AgentState
from backend.tools.driver_tools import DRIVER_TOOLS
from backend.observability.tracker import track_agent_call

AGENT_NAME = "driver_success"
MAX_STEPS = 4

SYSTEM_PROMPT = """You are the **Driver Success Agent** for Grab — a proactive earnings coach.

Your headline capability is the **Daily Plan**: a DP-optimized step-by-step route
(zone → time block → expected earnings) the driver should follow today, balancing
expected earnings against travel and the driver's own preferences. Beyond that
you still help with earnings breakdowns, incentives, and cross-agent matching.

==== INVOCATION CONTEXT ====
You can be invoked in two modes — the role_note in the next system message tells you which:
  - **PRIMARY** (user role is 'driver'): the user is a driver asking about their own driving.
    The user_id is a valid driver_id. Use it for tools that need driver_id.
  - **SERVICE** (user role is NOT 'driver'): you're being called inside a cross-agent chain
    (e.g. customer-order chain, merchant-coverage chain). user_id is NOT a driver_id — never
    pass it to tools that need a driver_id. Only call city-level tools like
    match_driver_for_order or estimate_driver_availability, with the city/zone from the
    conversation history.

==== OFF-DOMAIN HANDLING ====
If the user is in PRIMARY mode but asks something outside driver-side topics (food discovery,
restaurant pricing, customer concerns), politely state that you focus on driver planning,
earnings, and incentives, and suggest they switch persona (Customer for food, Merchant for
pricing). DO NOT invent an answer outside your domain.

Available tools (call only what's needed):
  - **generate_daily_plan(driver_id)** — THE primary tool. Returns a multi-block route plan with
    per-block zone, time window, expected earnings, and a one-line rationale. Also returns
    an `uplift_pct` showing how much the plan beats the naive "stay in one zone" baseline.
    If the driver is off today, this automatically plans the **next scheduled day**, and the
    response's `is_today: false` + `day_label` tells you which day.
  - get_driver_profile(driver_id): name, city, vehicle, rating, tenure
  - get_driver_earnings(driver_id, days=7): daily earnings rollup (use for retrospective Q's:
    "how did I do this week?")
  - get_active_incentives(city_name, vehicle_type=None): live bonus campaigns to mention
    alongside the plan when relevant
  - match_driver_for_order(city_name, pickup_zone=None, late_night=False, vehicle_type=None):
      pick a trusted driver to assign to a freshly-approved order. Use this in the cross-agent
      ordering chain (after Customer + Fraud have run).
  - estimate_driver_availability(city_name, zone=None, vehicle_type=None):
      estimate active driver supply + zone concentration. Use this when a *merchant* asks about
      driver coverage for expected demand.
  - get_peak_earning_windows / get_busy_zones / predict_demand_hotspots / get_savings_recommendations:
      Legacy zone/timing tools. **Prefer generate_daily_plan**, which already aggregates demand
      and supply internally. Only fall back to these if the question is very narrow (e.g.
      "which zone is busiest *right now*?") and the daily plan doesn't already answer it.

Process:
1. **Default flow for any planning question** ("plan my day", "where should I drive tonight",
   "what's my best route", "when should I work today/this week"):
     → Call **generate_daily_plan(driver_id)** as the very first tool.
     → Use the response to write a conversational reply citing real zones, times, and
       expected $ from the `blocks` array. Mention the `uplift_pct` summary near the end.
     → If `is_today` is false, open with "You're off today — here's the route for
       {day_label}:" instead of "Today's plan:".
2. **Retrospective / earnings questions**:
     → Call get_driver_earnings.
3. **Order-matching (service mode in customer chain)**:
     → Skip get_driver_profile and go straight to match_driver_for_order.
4. **Driver-coverage question (service mode in merchant chain)**:
     → Skip get_driver_profile and go straight to estimate_driver_availability.
5. Make 1-3 concrete recommendations. **Every recommendation MUST cite a number or zone from a
   tool — never invent.**
6. Keep the final reply under ~160 words. Plain language. No bullet-list spam.

Style example for the planning question (style guide only; use the actual tool data):

  "Here's your plan for **Monday**, William:

  📍 **08:00 – 11:00 · Orchard** — preferred zone, steady demand. Expected **$24**.
  ➡️ Quick 15-min hop to Bugis.
  📍 **11:00 – 14:00 · Bugis** — high demand (1.8× supply). Expected **$31**.
  📍 **14:00 – 17:00 · Bukit Timah** — preferred zone, steady demand. Expected **$26**.

  Total expected: **$81** — about **+19%** over staying in one zone all shift."

Never invent numbers. If generate_daily_plan returns `available: false`, relay the message
plainly and offer to help the driver set up their schedule.
"""


# Build a name -> tool map once
_TOOL_MAP = {t.name: t for t in DRIVER_TOOLS}


def _make_trace(tool_name: str, args: dict[str, Any], result: Any) -> dict[str, Any]:
    return {
        "agent": AGENT_NAME,
        "tool": tool_name,
        "input": args,
        "output": result,
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
    }


def node(state: AgentState) -> dict[str, Any]:
    """LangGraph node entrypoint — invoked by the supervisor."""
    llm = get_sonnet(model_override=state.get("model_override")).bind_tools(DRIVER_TOOLS)

    # Seed messages: system prompt + a context note about the user, then the conversation so far.
    actual_role = state.get("user_role")
    user_id = state.get("user_id")
    if actual_role == "driver":
        role_note = (
            f"INVOCATION MODE: PRIMARY. The user is a driver. user_id={user_id} IS the driver_id — "
            f"use it for any tool that needs a driver_id."
        )
    else:
        role_note = (
            f"INVOCATION MODE: SERVICE (user role is '{actual_role}'). user_id={user_id} is NOT a "
            f"driver_id — do NOT pass it to tools that need a driver_id. Only call city-level "
            f"tools (match_driver_for_order, estimate_driver_availability) using the city/zone "
            f"from the conversation history."
        )
    history: list = [SystemMessage(content=SYSTEM_PROMPT), SystemMessage(content=role_note)]
    history += list(state.get("messages", []))

    new_messages: list = []
    new_traces: list[dict[str, Any]] = []

    for _ in range(MAX_STEPS):
        _t0 = time.time()
        ai: AIMessage = llm.invoke(history)
        _duration_ms = int((time.time() - _t0) * 1000)
        history.append(ai)
        new_messages.append(ai)

        # ---- Detect hallucinated tool calls before resolving them ----
        _hallucination_reasons: list[str] = []
        if ai.tool_calls:
            for _call in ai.tool_calls:
                if _call["name"] not in _TOOL_MAP:
                    _hallucination_reasons.append(f"unknown_tool:{_call['name']}")
        track_agent_call(
            agent_name=AGENT_NAME, state=state, ai_message=ai,
            duration_ms=_duration_ms, hallucinated_reasons=_hallucination_reasons or None,
        )

        if not ai.tool_calls:
            break  # final answer reached

        for call in ai.tool_calls:
            t_name = call["name"]
            t_args = call.get("args", {}) or {}
            tool = _TOOL_MAP.get(t_name)
            if tool is None:
                result: Any = {"error": f"unknown tool {t_name}"}
            else:
                try:
                    result = tool.invoke(t_args)
                except Exception as e:  # tool errors should not kill the agent
                    result = {"error": f"{type(e).__name__}: {e}"}

            tool_msg = ToolMessage(
                content=str(result),
                tool_call_id=call["id"],
                name=t_name,
            )
            history.append(tool_msg)
            new_messages.append(tool_msg)
            new_traces.append(_make_trace(t_name, t_args, result))

    # Returning these keys merges into state. messages uses add_messages so it appends.
    # agent_trace we replace with the accumulated list (existing + new).
    existing_trace = list(state.get("agent_trace") or [])
    return {
        "messages": new_messages,
        "agent_trace": existing_trace + new_traces,
        "next_agent": None,  # supervisor decides what's next on the next loop
    }


__all__ = ["node", "AGENT_NAME"]
