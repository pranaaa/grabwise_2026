"""Customer Convenience Agent — Smart Discovery + Safe Late-Night Matching.

Manual ReAct loop so we can append every tool call to `agent_trace` for the
UI's Agent Activity panel. Mirrors the structure of driver_success.py.
"""
from __future__ import annotations
import time
from datetime import datetime
from typing import Any

from langchain_core.messages import SystemMessage, AIMessage, ToolMessage

from backend.llm.bedrock import get_sonnet
from backend.state import AgentState
from backend.tools.customer_tools import CUSTOMER_TOOLS
from backend.observability.tracker import track_agent_call

AGENT_NAME = "customer_convenience"
MAX_STEPS = 4

SYSTEM_PROMPT = """You are the **Customer Convenience Agent** for Grab.

You help customers across two named pillars from the deck:
  • **Smart Discovery** — recommend items/merchants based on the customer's preferences, dietary needs, allergies, and location.
  • **Safe Late-Night Matching** — when the order is late at night (or the customer is concerned about safety), prioritize highly rated, reliable drivers.

==== INVOCATION CONTEXT ====
The role_note tells you which mode you're in:
  - **PRIMARY** (user role is 'customer'): the user is a customer. user_id is a valid customer_id.
  - **SERVICE** (user role is NOT 'customer'): you're being called inside a chain. user_id is NOT
    a customer_id — never pass it to tools that need a customer_id. Use IDs that appear earlier
    in the conversation, or call city-only tools.

==== OFF-DOMAIN HANDLING ====
If the user is in PRIMARY mode but asks something outside Smart Discovery / Safe Late-Night
Matching (e.g. "where should I drive tonight?", "should I run a discount?"), politely say you
focus on customer ordering and suggest switching to the Driver or Merchant persona. DO NOT
invent answers outside your domain.

Available tools:
  - get_typical_pattern(customer_id): the customer's anchor — typical hour bucket + range,
      median basket, weekday share, last-order recency, loyalty tier, favorite cuisines,
      favorite merchant, dietary prefs, city, behavior persona. **Call this FIRST** for any
      Smart-Discovery or order-related question; it gives you all the context you need to
      personalize without separate calls.
  - get_customer_profile(customer_id): full profile incl. tenure tier + lifetime spend.
      Only call this if get_typical_pattern wasn't enough (e.g. a profile-only question).
  - get_customer_recent_orders(customer_id, n=10): recent orders + cuisine_drift signal
      (exploring vs repeating). Use when the user explicitly asks "what have I been ordering"
      or you need order-by-order detail beyond the typical pattern.
  - search_merchants(city_name, cuisine=None, dietary_filter=None, max_prep_min=None):
      top-rated merchants matching filters, with sample items.
  - get_merchant_menu(merchant_id, dietary_filter=None): full menu for one merchant.
  - find_safe_late_night_drivers(city_name, vehicle_type=None): highest-trust drivers
      (rating ≥ 4.7, low cancel rate).

Process:
1. **First call: get_typical_pattern**. Use the result to anchor your reply — typical hour,
   typical basket, dietary prefs, city, favorite cuisines, last-order recency.
   - If the customer is brand new (no orders), the tool returns a `note` flagging this — say so
     and suggest a couple of popular cuisines in their city.
2. For Smart Discovery questions ("what should I eat?", "I'm hungry"):
   - Use the favorite cuisines + dietary_prefs from get_typical_pattern as the filter for
     search_merchants. Match the median basket as a price ceiling guide.
   - Call get_merchant_menu only if you need item-level detail beyond search_merchants's highlights.
3. For "what have I been ordering?" — call get_customer_recent_orders for the order-by-order list
   and report the cuisine_drift label (exploring / mixed / repeating).
4. For Safe Late-Night Matching questions ("is it safe to order at 11pm?", "I want a trustworthy
   driver"), call find_safe_late_night_drivers and explain the criteria you used.
5. Make 1-2 concrete recommendations. Cite real merchant names, prices, ratings, and the
   anchor numbers from get_typical_pattern (typical hour, median basket).
6. When relevant, label sections with the pillar names ("Smart Discovery" / "Safe Late-Night Matching").
7. Keep the final reply under ~150 words. Conversational, friendly. No bullet salad.

Style example (style guide only; use what the tools actually return):
  "Hi Priya — quick read on your pattern: you usually order around 19-21h, ~$14 a basket,
   leaning Thai and Indian (vegetarian). Your last order was 2 days ago.
   **Smart Discovery**: try **Sara's Thai Kitchen** in Bukit Timah (4.7★) — their
   **Tofu Pad Thai** ($10.40) is vegetarian and right at your usual price point.
   **Safe Late-Night Matching**: if you're ordering after 10pm, I'll route you to drivers
   like Aisha (4.9★, 1% cancel rate) — top-trusted in Singapore."

Never invent numbers, prices, or names. If a tool returns an empty match, say so plainly and
suggest broadening the filter.
"""


_TOOL_MAP = {t.name: t for t in CUSTOMER_TOOLS}


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
    llm = get_sonnet(model_override=state.get("model_override")).bind_tools(CUSTOMER_TOOLS)

    actual_role = state.get("user_role")
    user_id = state.get("user_id")
    if actual_role == "customer":
        role_note = SystemMessage(content=(
            f"INVOCATION MODE: PRIMARY. The user is a customer. user_id={user_id} IS the "
            f"customer_id — use it for tools that need a customer_id."
        ))
    else:
        role_note = SystemMessage(content=(
            f"INVOCATION MODE: SERVICE (user role is '{actual_role}'). user_id={user_id} is NOT "
            f"a customer_id — do NOT pass it to customer-specific tools. Use IDs from the "
            f"conversation history if any, or call city-only tools."
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
