"""Tools exposed to the Planner agent.

  - spawn_specialist : create an ephemeral worker for a focused sub-task
  - list_available_tools : enumerate the tool registry so the planner knows
                           what subsets are possible

Both are @tool-decorated so they're callable through LangChain's tool-binding
flow — the planner emits a structured tool_call, our agent loop invokes the
underlying function, the response goes back into context.

Per-run spawn cap
-----------------
A ContextVar counter (spawn_count_var) is reset at the start of each Planner
node() invocation. spawn_specialist_tool increments it and refuses with a
clear message once MAX_SPAWNS_PER_RUN is hit. This protects against runaway
spawning if the LLM gets confused.
"""
from __future__ import annotations
from contextvars import ContextVar
from typing import Any

from langchain_core.tools import tool

from backend.agents.spawn import spawn_specialist
from backend.tools.registry import list_tool_names_by_group, ALL_TOOLS

MAX_SPAWNS_PER_RUN = 4

# Reset to 0 at the top of every Planner node() invocation.
spawn_count_var: ContextVar[int] = ContextVar("spawn_count", default=0)


@tool
def spawn_specialist_tool(
    name: str,
    system_prompt: str,
    tool_names: list[str],
    query: str,
) -> dict[str, Any]:
    """Create and run an ephemeral specialist agent for one focused sub-task.

    Use this when the user's question is multi-dimensional and you need to
    delegate part of it. Each spawn is independent — give it a tight system
    prompt, a minimal tool subset, and a single sub-question to answer.

    Args:
        name: Short identifier for telemetry (e.g. "sg_late_night_drivers").
        system_prompt: Detailed focused instructions for the specialist —
            what the sub-task is, what to return, what NOT to do.
        tool_names: Subset of tool names from the registry. Call
            list_available_tools first if you're unsure what's available.
        query: The single sub-question for the specialist.

    Returns:
        {
          "name": str,
          "answer": str,           # the specialist's final reply
          "steps_used": int,
          "tools_called": list[str],
          "hallucinated": bool,
        }

    Hard limit: at most 4 spawns per Planner run. Beyond that this tool
    returns an error dict and you should synthesize from what you have.
    """
    n = spawn_count_var.get()
    if n >= MAX_SPAWNS_PER_RUN:
        return {
            "error": f"spawn budget exhausted ({MAX_SPAWNS_PER_RUN} spawns used) — "
                     f"synthesize from existing results without spawning more",
            "spawns_used": n,
        }
    spawn_count_var.set(n + 1)
    return spawn_specialist(
        name=name,
        system_prompt=system_prompt,
        tool_names=tool_names,
        query=query,
    )


@tool
def list_available_tools() -> dict[str, Any]:
    """Enumerate every tool a spawned specialist can be given access to.

    Returns:
        {
          "groups": {
             "driver":   ["get_driver_profile", ...],
             "customer": ["get_customer_profile", ...],
             "merchant": ["get_merchant_profile", ...],
             "fraud":    ["score_order_risk", ...],
          },
          "total": int
        }

    Use this when planning sub-tasks to pick the minimal viable toolkit
    for each specialist.
    """
    groups = list_tool_names_by_group()
    return {"groups": groups, "total": len(ALL_TOOLS)}


PLANNER_TOOLS = [spawn_specialist_tool, list_available_tools]


def reset_spawn_counter() -> None:
    """Call this at the top of each Planner node() invocation."""
    spawn_count_var.set(0)
