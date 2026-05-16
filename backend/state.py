"""LangGraph shared state.

`messages` accumulates the conversation (supervisor + agents append AIMessages).
`agent_trace` is what the future Agent Activity Panel reads — every tool call
appends an entry so the UI can animate.
"""
from __future__ import annotations
from typing import TypedDict, Annotated, Literal, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


UserRole = Literal["driver", "customer", "merchant", "admin"]


class TraceEntry(TypedDict, total=False):
    agent: str
    tool: str
    input: dict[str, Any]
    output: Any
    ts: str


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    user_role: UserRole
    user_id: str | None
    next_agent: str | None
    agent_trace: list[TraceEntry]
    # Optional Bedrock model_id picked by the user via the chat dropdown.
    # None → factory uses the .env defaults. Threaded into get_sonnet/get_haiku.
    model_override: str | None
