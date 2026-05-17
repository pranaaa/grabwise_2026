"""Planner Agent — decomposes complex multi-dimensional queries.

When the user's question spans multiple personas, cities, time windows,
or analytical dimensions, the supervisor routes here. The Planner:

  1. Decomposes the query into 2-4 focused sub-tasks
  2. Calls spawn_specialist for each — picking a tight system prompt and
     a minimal tool subset per sub-task
  3. Synthesizes the spawn outputs into a single coherent reply

Why a planner instead of more specialists?
------------------------------------------
Adding a new specialist agent for every novel question shape doesn't scale.
The Planner can compose a fresh worker tailored to any sub-task at runtime
using tools drawn from the existing registry. New capabilities emerge from
combinations of existing tools rather than new code.

Budgets
-------
  - MAX_STEPS (Planner ReAct loop)  = 6
  - MAX_SPAWN_STEPS (each spawn)    = 3 (set in spawn.py)
  - MAX_SPAWNS_PER_RUN              = 4 (set in planner_tools.py)
"""
from __future__ import annotations
import time
from datetime import datetime
from typing import Any

from langchain_core.messages import SystemMessage, AIMessage, ToolMessage

from backend.llm.bedrock import get_sonnet
from backend.state import AgentState
from backend.tools.planner_tools import PLANNER_TOOLS, reset_spawn_counter
from backend.observability.tracker import track_agent_call
from backend.agents.spawn import current_model_override

AGENT_NAME = "planner"
MAX_STEPS = 6


SYSTEM_PROMPT = """You are the **Planner Agent** for GrabWise — an orchestrator
for analytical questions that span multiple personas, cities, time windows,
or data dimensions.

When to use you
---------------
You are routed to for questions like:
  - "Compare X across cities/segments/time windows"
  - "Give me a strategy memo on..."
  - "Deep dive into how X relates to Y for Z customers"
  - Cross-persona analysis (driver + customer + merchant in one ask)
  - Novel multi-step analyses that don't fit a single specialist

You are NOT for simple single-dimensional questions ("what's my plan today",
"forecast my demand Friday") — those go directly to specialists.

How to operate
--------------
1. **Plan first.** Read the user's question. Decompose it into 2–4 focused
   sub-tasks. Each sub-task should be answerable by ONE specialist with a
   SMALL toolkit.

2. **(Optional) Check available tools.** Call list_available_tools to see
   what tool names are registered. Skip this if you already know what you
   need.

3. **Spawn workers.** For each sub-task, call spawn_specialist with:
     - name           : short snake_case id (e.g. "sg_late_night_drivers")
     - system_prompt  : tight, focused, includes WHAT to produce + DON'T-DO
     - tool_names     : ONLY the tools that worker needs (driver_*, customer_*,
                        merchant_*, fraud_*). Stay minimal.
     - query          : the SINGLE sub-question for the worker

4. **Synthesize.** After all spawns return, write ONE coherent reply that
   integrates their findings. Cite which spawn produced which data so the
   user can trace it.

Hard rules
----------
- Maximum 4 spawns per request. Plan accordingly.
- Each spawn has a 3-step budget — give it a TIGHT focused sub-question.
- NEVER invent data. Only synthesize from what spawns actually returned.
- If a spawn errors or returns nothing useful, mention it openly in the reply
  rather than papering over it.

Output style
------------
Brief and structured. Default format:

  **Headline** — one-line takeaway

  **Findings**
  - [from {spawn_name}] Fact + relevant numbers
  - [from {spawn_name}] Fact + relevant numbers
  - ...

  **Bottom line** — one-line recommendation

Keep the final reply under ~300 words. No bullet-list spam beyond the
findings block.

How to write GOOD spawn prompts
-------------------------------
A spawn has only 6 ReAct steps and a small toolkit. Give it a TIGHT,
specific sub-question with hints about which tools to call in what order.

❌ BAD spawn prompt (too vague — spawn will flounder):
   query: "Find driver earnings for vegetarian customers in Singapore at late night"
   system_prompt: "You are a data analyst. Answer the question."

✅ GOOD spawn prompt (specific + tool-sequenced):
   tool_names: ["search_merchants", "find_safe_late_night_drivers",
                "get_busy_zones", "predict_demand_hotspots"]
   system_prompt: |
     You analyze Singapore's late-night (10pm-2am) demand for vegetarian
     food. Process:
       1. Call search_merchants(city_name="Singapore", dietary_filter="vegetarian")
          to find vegetarian-friendly merchants.
       2. Call get_busy_zones(city_name="Singapore", hour=22) to identify
          where late-night activity concentrates.
       3. Call find_safe_late_night_drivers(city_name="Singapore") to see
          which drivers serve this window.
     Return ONE paragraph: (a) how many vegetarian merchants exist,
     (b) which zones are busiest 10pm-2am, (c) typical late-night driver
     trust score. No invented numbers — cite the tool you got each fact from.
   query: "Profile Singapore's vegetarian late-night ecosystem"

Always:
  - List the exact tools the spawn should call
  - Give numbered steps when a sequence matters
  - Specify the OUTPUT shape (a paragraph? a list? bullet of metrics?)
  - Forbid invented numbers explicitly

Example query → plan → synthesis
---------------------------------
Query: "Compare driver earnings in Singapore vs Jakarta for late-night
orders by vegetarian customers — give me a strategy memo."

Plan:
  spawn 1: "sg_veg_late_night"
    tools: ["search_merchants", "get_busy_zones", "find_safe_late_night_drivers"]
    query: "Profile Singapore's vegetarian late-night supply + demand"
  spawn 2: "jakarta_veg_late_night"  (same toolset, swap city)
  spawn 3: "cross_city_synthesis"
    tools: []  (no tools — pure reasoning over spawn-1 and spawn-2 results which you'll pass in)
    query: "Given the two city profiles below, identify the bigger
            opportunity and 2 actionable differences"

Then write the memo: headline / findings / bottom line.
"""


def node(state: AgentState) -> dict[str, Any]:
    """LangGraph entrypoint — the Planner's ReAct loop."""

    # 1. Reset per-run spawn counter (a ContextVar guard against runaway spawning)
    reset_spawn_counter()

    # 2. Make sure spawned specialists inherit the user's model_override
    cv_token = current_model_override.set(state.get("model_override"))

    try:
        llm = get_sonnet(model_override=state.get("model_override")).bind_tools(PLANNER_TOOLS)
        tool_map = {t.name: t for t in PLANNER_TOOLS}

        history: list = [SystemMessage(content=SYSTEM_PROMPT)]
        history += list(state.get("messages", []))

        new_messages: list = []
        new_traces: list[dict[str, Any]] = []

        for _ in range(MAX_STEPS):
            t0 = time.time()
            ai: AIMessage = llm.invoke(history)
            duration_ms = int((time.time() - t0) * 1000)
            history.append(ai)
            new_messages.append(ai)

            # Detect hallucinated tool calls (planner only has 2 tools)
            reasons: list[str] = []
            if ai.tool_calls:
                for call in ai.tool_calls:
                    if call["name"] not in tool_map:
                        reasons.append(f"unknown_tool:{call['name']}")
            track_agent_call(
                agent_name=AGENT_NAME, state=state, ai_message=ai,
                duration_ms=duration_ms, hallucinated_reasons=reasons or None,
            )

            if not ai.tool_calls:
                # Final synthesis — exit the loop
                break

            for call in ai.tool_calls:
                t_name = call["name"]
                t_args = call.get("args", {}) or {}
                tool = tool_map.get(t_name)
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
                new_traces.append({
                    "agent": AGENT_NAME,
                    "tool": t_name,
                    "input": t_args,
                    "output": result,
                    "ts": datetime.utcnow().isoformat(timespec="seconds"),
                })

        existing_trace = list(state.get("agent_trace") or [])
        return {
            "messages": new_messages,
            "agent_trace": existing_trace + new_traces,
            "next_agent": None,
        }
    finally:
        current_model_override.reset(cv_token)


__all__ = ["node", "AGENT_NAME"]
