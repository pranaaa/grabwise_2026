"""POST /api/chat — streams a GrabWise supervisor run as Server-Sent Events.

Event types emitted to the client:
  - supervisor_decision : { next_agent }                       (per supervisor turn)
  - tool_call           : { agent, tool, input, output, ts }   (per tool result)
  - agent_message       : { content, agent }                   (one per substantive AI reply)
  - done                : {}
  - error               : { message }
"""
from __future__ import annotations
import json
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage, AIMessage

from backend.api.schemas import ChatRequest
from backend.api.auth import get_current_user, CurrentUser
from backend.agents.supervisor import GRAPH
from backend.db.database import get_session
from backend.db import models as M
from backend.llm.registry import is_valid as model_is_valid

router = APIRouter(prefix="/api/chat", tags=["chat"])


# Defense-in-depth: which agents may be the *first* (entry) agent for each role.
ROLE_ENTRY_ALLOWED = {
    "driver":   {"driver_success", "planner"},
    "customer": {"customer_convenience", "planner"},
    "merchant": {"merchant_growth", "planner"},
    "admin":    {"driver_success", "customer_convenience", "merchant_growth", "fraud_risk", "planner"},
}
ROLE_DEFAULT_AGENT = {
    "driver":   "driver_success",
    "customer": "customer_convenience",
    "merchant": "merchant_growth",
    "admin":    "driver_success",
}


def _is_substantive(msg: AIMessage) -> bool:
    text = msg.content if isinstance(msg.content, str) else ""
    if not text:
        return False
    if text.startswith("[supervisor]"):
        return False
    if getattr(msg, "tool_calls", None):
        return False
    return True


def _persist_message(auth_user_id: int, role: str, content: str, agent: str | None = None) -> None:
    try:
        with get_session() as s:
            s.add(M.ChatMessage(
                auth_user_id=auth_user_id,
                role=role,
                agent=agent,
                content=content[:8000],
            ))
    except Exception:
        pass


@router.post("")
async def chat(req: ChatRequest, current: CurrentUser = Depends(get_current_user)):
    role = current.role
    user_id = str(current.id)

    # Only honour model_override if it matches a registry entry — guards against
    # arbitrary values being shoved into the Bedrock client.
    model_override = req.model if model_is_valid(req.model) else None

    initial_state = {
        "messages": [HumanMessage(content=req.message)],
        "user_role": role,
        "user_id": user_id,
        "agent_trace": [],
        "next_agent": None,
        "model_override": model_override,
    }

    _persist_message(current.auth_user_id, "user", req.message)

    async def stream() -> AsyncGenerator[dict, None]:
        seen_traces = 0
        seen_msgs = 0
        last_routed: str | None = None
        last_agent_for_msg: str | None = None
        first_decision_checked = False
        allowed_entry = ROLE_ENTRY_ALLOWED.get(role, set())

        try:
            async for state in GRAPH.astream(
                initial_state,
                stream_mode="values",
                config={"recursion_limit": 25},
            ):
                routed = state.get("next_agent")
                if routed and routed != last_routed:
                    # Defense-in-depth: rewrite the very first entry agent if the
                    # supervisor returned something not allowed for this role.
                    if not first_decision_checked and routed not in {"FINISH", None}:
                        first_decision_checked = True
                        if routed not in allowed_entry:
                            print(f"[chat] role={role} blocked entry agent {routed!r}; "
                                  f"rerouting to {ROLE_DEFAULT_AGENT[role]!r}")
                            routed = ROLE_DEFAULT_AGENT[role]
                            state["next_agent"] = routed
                    last_routed = routed
                    last_agent_for_msg = routed
                    yield {
                        "event": "supervisor_decision",
                        "data": json.dumps({"next_agent": routed}),
                    }

                trace = state.get("agent_trace") or []
                for entry in trace[seen_traces:]:
                    yield {
                        "event": "tool_call",
                        "data": json.dumps(entry, default=str),
                    }
                seen_traces = len(trace)

                messages = state.get("messages", [])
                for msg in messages[seen_msgs:]:
                    if isinstance(msg, AIMessage) and _is_substantive(msg):
                        agent_name = last_agent_for_msg or "agent"
                        text = msg.content if isinstance(msg.content, str) else str(msg.content)
                        _persist_message(current.auth_user_id, "assistant", text, agent_name)
                        yield {
                            "event": "agent_message",
                            "data": json.dumps({
                                "content": msg.content,
                                "agent": agent_name,
                            }),
                        }
                seen_msgs = len(messages)

            yield {"event": "done", "data": "{}"}

        except Exception as e:
            # Stringify carefully. Bare KeyError(s) stringify to "'FINISH'" which
            # is opaque to end-users — wrap in something readable and suggest
            # a remedy if it looks like a model-protocol slip.
            raw = str(e) or e.__class__.__name__
            err_kind = e.__class__.__name__
            if err_kind in {"KeyError", "ValidationError"}:
                pretty = (
                    f"The selected model returned an unexpected response "
                    f"({err_kind}: {raw}). Try switching to a ★ recommended "
                    f"model (Qwen3 235B or DeepSeek V3.2) for more reliable "
                    f"tool calling."
                )
            else:
                pretty = f"{err_kind}: {raw}"
            yield {"event": "error", "data": json.dumps({"message": pretty})}

    return EventSourceResponse(stream())
