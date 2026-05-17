"""Central registry of every tool available across the four specialist agents.

Used by the Planner agent's spawn_specialist runtime: when the planner
decomposes a complex query into sub-tasks, it picks a tool subset (by name)
from this registry to bind to each ephemeral specialist.

Source of truth for which tools exist in the system. If a new tool is added
to any agent's *_tools.py file (and exported in *_TOOLS), it appears here
automatically — no manual registration needed.

Tool-name collisions across agents are flagged on import so we catch them
early rather than silently picking one.
"""
from __future__ import annotations
from typing import Any

from backend.tools.driver_tools   import DRIVER_TOOLS
from backend.tools.customer_tools import CUSTOMER_TOOLS
from backend.tools.merchant_tools import MERCHANT_TOOLS
from backend.tools.fraud_tools    import FRAUD_TOOLS


# Per-group lists — useful when the planner wants "all driver tools"
TOOL_GROUPS: dict[str, list[Any]] = {
    "driver":   list(DRIVER_TOOLS),
    "customer": list(CUSTOMER_TOOLS),
    "merchant": list(MERCHANT_TOOLS),
    "fraud":    list(FRAUD_TOOLS),
}

# Flat dict: tool_name → tool object. Built once at import; warn on dup names.
ALL_TOOLS: dict[str, Any] = {}
_collisions: list[tuple[str, str]] = []   # (tool_name, second_group)
for _group, _tools in TOOL_GROUPS.items():
    for _t in _tools:
        if _t.name in ALL_TOOLS:
            _collisions.append((_t.name, _group))
        ALL_TOOLS[_t.name] = _t

if _collisions:
    import logging
    logging.getLogger(__name__).warning(
        "Tool-name collisions in registry — last-write wins: %s",
        _collisions,
    )


# Public helpers --------------------------------------------------------------
def get_tools(names: list[str]) -> list[Any]:
    """Resolve a list of tool names to live tool objects.

    Unknown names are silently dropped (with a logger warning) so a planner
    spawn with one bad name still gets the rest of its toolkit.
    """
    out: list[Any] = []
    missing: list[str] = []
    for n in names or []:
        t = ALL_TOOLS.get(n)
        if t is None:
            missing.append(n)
        else:
            out.append(t)
    if missing:
        import logging
        logging.getLogger(__name__).debug("Registry skipped unknown tools: %s", missing)
    return out


def list_tool_names_by_group() -> dict[str, list[str]]:
    """Return {group: [tool_name, ...]} — for surfacing in API or planner prompt."""
    return {g: [t.name for t in tools] for g, tools in TOOL_GROUPS.items()}


def all_tool_names() -> list[str]:
    return list(ALL_TOOLS.keys())
