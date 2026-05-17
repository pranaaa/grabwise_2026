"""Ephemeral specialist agent runtime — the engine behind the Planner.

The Planner agent calls `spawn_specialist(...)` (via a LangChain tool wrapper)
to delegate a focused sub-task to a fresh ad-hoc worker. We construct a
ReAct loop on the fly, bound to a custom system prompt + tool subset, run
it for at most MAX_SPAWN_STEPS, and return the final text answer.

Why ephemeral?
--------------
For one-off sub-questions, having a permanent agent in the LangGraph is
overkill — and adds prompt-tuning friction. Spawning lets the Planner shape
a worker exactly for the sub-task: tighter prompt, smaller tool surface,
lower hallucination odds.

Threading the user's model selection
-------------------------------------
The chat dropdown's model_override lives in the AgentState. The Planner's
node() sets a ContextVar before invoking the LLM; spawn_specialist reads
that ContextVar so every spawn uses the same model the user picked. No
extra plumbing through the LangChain tool boundary.

Budget caps (per spawn)
-----------------------
- MAX_SPAWN_STEPS = 3   — total ReAct iterations
- Cost is tracked per call in llm_call_logs with agent="spawn:<name>"

Budget caps (per Planner run)
-----------------------------
Enforced inside the LangChain tool wrapper in planner_tools.py via a
ContextVar counter, default 4 spawns / Planner invocation.
"""
from __future__ import annotations
import time
from contextvars import ContextVar
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from backend.llm.bedrock import get_sonnet
from backend.tools.registry import get_tools
from backend.observability.tracker import track_agent_call

# Per-spawn ReAct iteration cap. 6 lets a spawn do a meaningful multi-tool
# analysis (e.g. profile → recent_orders → search_merchants → menu → safety
# → synthesis). At 3 the spawn frequently exhausted its budget after a
# single discovery call and couldn't actually answer the sub-question.
MAX_SPAWN_STEPS = 6

# Threaded by the Planner's node() so spawned specialists inherit the chat's
# model_override (the user's dropdown choice). Default None → factory uses .env.
current_model_override: ContextVar[str | None] = ContextVar(
    "current_model_override", default=None
)


def spawn_specialist(
    *,
    name: str,
    system_prompt: str,
    tool_names: list[str],
    query: str,
) -> dict[str, Any]:
    """Run an ephemeral ReAct agent to completion (or step cap).

    Args:
        name: Short identifier for telemetry — e.g. "sg_driver_late_night".
              Surfaces as agent="spawn:<name>" in llm_call_logs.
        system_prompt: Detailed instructions for THIS spawn's sub-task only.
        tool_names: Subset of registered tool names this spawn can call.
                    Unknown names are silently dropped (logged in registry).
        query: The focused sub-question. Goes in as a HumanMessage.

    Returns dict:
        name                  : echo of input
        answer                : final text from the spawn
        steps_used            : ReAct iterations consumed
        tools_called          : list of tool names invoked
        hallucinated          : bool — any unknown-tool or exception
        hallucination_reasons : list[str] — structured reason codes
    """
    model_override = current_model_override.get()
    tools = get_tools(tool_names)
    tool_map = {t.name: t for t in tools}

    llm = get_sonnet(model_override=model_override)
    if tools:
        llm = llm.bind_tools(tools)

    history: list = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]

    tools_called: list[str] = []
    hallucination_reasons: list[str] = []
    answer: str | None = None
    steps_used = 0

    for step in range(MAX_SPAWN_STEPS):
        t0 = time.time()
        try:
            ai: AIMessage = llm.invoke(history)
        except Exception as e:
            hallucination_reasons.append(f"invoke_exception:{type(e).__name__}")
            answer = f"Spawn {name!r} failed: {type(e).__name__}: {e}"
            steps_used = step + 1
            # Log this failed call too
            track_agent_call(
                agent_name=f"spawn:{name}",
                state={"model_override": model_override},
                ai_message=None,
                duration_ms=int((time.time() - t0) * 1000),
                hallucinated_reasons=hallucination_reasons,
            )
            break

        duration_ms = int((time.time() - t0) * 1000)
        history.append(ai)
        steps_used = step + 1

        # ----- Resolve any tool calls before tracking the step -----
        step_reasons: list[str] = []
        if ai.tool_calls:
            for call in ai.tool_calls:
                t_name = call["name"]
                t_args = call.get("args", {}) or {}
                tools_called.append(t_name)
                tool = tool_map.get(t_name)
                if tool is None:
                    step_reasons.append(f"unknown_tool:{t_name}")
                    result: Any = {
                        "error": f"unknown tool {t_name!r} — not in spawn's tool subset"
                    }
                else:
                    try:
                        result = tool.invoke(t_args)
                    except Exception as e:
                        result = {"error": f"{type(e).__name__}: {e}"}
                history.append(ToolMessage(
                    content=str(result),
                    tool_call_id=call["id"],
                    name=t_name,
                ))

        # Telemetry for this LLM call inside the spawn
        track_agent_call(
            agent_name=f"spawn:{name}",
            state={"model_override": model_override},
            ai_message=ai,
            duration_ms=duration_ms,
            hallucinated_reasons=step_reasons or None,
        )
        hallucination_reasons.extend(step_reasons)

        # If no tool calls, the model produced its final answer
        if not ai.tool_calls:
            answer = ai.content if isinstance(ai.content, str) else str(ai.content)
            break

    if answer is None:
        # Step budget exhausted without a final reply — take the last AI content
        last_ai = next(
            (m for m in reversed(history) if isinstance(m, AIMessage)), None,
        )
        answer = (
            (last_ai.content if isinstance(last_ai.content, str) else str(last_ai.content))
            if last_ai
            else f"(spawn {name!r} hit step budget without producing an answer)"
        )

    return {
        "name": name,
        "answer": answer,
        "steps_used": steps_used,
        "tools_called": tools_called,
        "hallucinated": bool(hallucination_reasons),
        "hallucination_reasons": hallucination_reasons,
    }
