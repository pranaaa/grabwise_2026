"""Fraud & Risk Agent — Driver Trust, Late-Night Matching, Customer Anomaly Detection.

Manual ReAct loop bound to the fraud tools. In the cross-agent ordering chain,
this agent reads the customer + (optional) order context from message history,
scores the order risk, and lets the supervisor decide whether to proceed to
driver assignment.
"""
from __future__ import annotations
import time
from datetime import datetime
from typing import Any

from langchain_core.messages import SystemMessage, AIMessage, ToolMessage

from backend.llm.bedrock import get_sonnet
from backend.state import AgentState
from backend.tools.fraud_tools import FRAUD_TOOLS
from backend.observability.tracker import track_agent_call

AGENT_NAME = "fraud_risk"
MAX_STEPS = 4

SYSTEM_PROMPT = """You are the **Fraud & Risk Agent** for Grab.

You cover three named pillars from the deck:
  • **Driver Trust Scoring** — score using safety feedback, reliability, complaint history.
  • **Trusted Late-Night Matching** — confirm driver trust is high enough for late-night orders.
  • **Customer Anomaly Detection** — flag unusual customer behaviour (orders, cancels, account age).

Available tools:
  - score_driver_trust(driver_id) → 0-100, higher = more trusted, with reasons
  - score_customer_anomaly(customer_id) → 0-100, higher = more suspicious, with flags
  - score_order_risk(customer_id, estimated_total=None, late_night=False) → 0-100 + decision (approve/review/block)
  - get_transaction_signals(order_id) → details for a specific past order

Process:
1. If the user just asked for a "risk check" / "is this safe" / is placing an order, call **score_order_risk** with the customer_id (= user_id in state). Pass `late_night=True` if the conversation mentions late hours.
2. If they explicitly ask about a specific driver, call **score_driver_trust**.
3. If they ask about a specific past transaction, call **get_transaction_signals** then optionally **score_customer_anomaly**.
4. Always include the numerical score in your final reply, plus 1-2 short reasons drawn from the tool's `reasons`/`flags`/`contributions`.
5. If this is a step in an order-placement chain (recognizable because the previous assistant message proposed an order), explicitly say "approving for driver matching" or "flagging for review" so the supervisor knows what to do next.

Format:
  - Concise, analytical. Under ~110 words.
  - Label sections with pillar names where natural ("Order Risk Check" / "Driver Trust" / "Anomaly Check").
  - Never invent numbers — every score in your reply must come from a tool result.
"""


_TOOL_MAP = {t.name: t for t in FRAUD_TOOLS}


def _make_trace(tool_name: str, args: dict[str, Any], result: Any) -> dict[str, Any]:
    return {
        "agent": AGENT_NAME,
        "tool": tool_name,
        "input": args,
        "output": result,
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
    }


def node(state: AgentState) -> dict[str, Any]:
    llm = get_sonnet(model_override=state.get("model_override")).bind_tools(FRAUD_TOOLS)

    actual_role = state.get("user_role")
    user_id = state.get("user_id")
    if actual_role == "customer":
        role_note = SystemMessage(content=(
            f"INVOCATION CONTEXT: the user is a customer. user_id={user_id} IS the customer_id — "
            f"use it for score_customer_anomaly / score_order_risk. For driver lookups, use a "
            f"driver_id mentioned in the conversation."
        ))
    else:
        role_note = SystemMessage(content=(
            f"INVOCATION CONTEXT: the user's role is '{actual_role}', not 'customer'. user_id="
            f"{user_id} is NOT a customer_id. Only call score_customer_anomaly / score_order_risk "
            f"with a customer_id that appears in the conversation. score_driver_trust expects a "
            f"driver_id from the conversation. If neither applies, decline politely."
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
