"""GrabWise Supervisor — routes user requests to the right specialist agent.

Built around LangGraph's supervisor pattern: every specialist agent ends with an
edge back to the supervisor, which then either dispatches to another agent or
returns FINISH to end the run.

Right now only `driver_success` is implemented; the other agents are stubs that
politely say "coming soon." This keeps the routing topology stable so we can
swap real implementations in without touching the graph.
"""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

import time

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, END

from backend.llm.bedrock import get_haiku
from backend.state import AgentState
from backend.agents import driver_success, customer_convenience, merchant_growth, fraud_risk, planner
from backend.observability.tracker import track_agent_call


# Update this Literal as new agents come online.
AgentName = Literal[
    "driver_success",
    "customer_convenience",
    "merchant_growth",
    "fraud_risk",
    "planner",
    "FINISH",
]


class RouteDecision(BaseModel):
    """Structured decision returned by the supervisor LLM."""
    next_agent: AgentName = Field(description="Which agent should run next, or FINISH if the user's request is fully answered.")
    reason: str = Field(description="One short sentence explaining the choice.")


SUPERVISOR_PROMPT = """You are the GrabWise Supervisor — the orchestrator for a multi-agent system serving Grab's ecosystem (drivers, customers, merchants).

You decide which specialist agent runs next, or whether the user's request has been fully answered.

Available specialist agents:
- driver_success: earnings guidance, busy zones, incentives, AND driver-matching for orders
- customer_convenience: meal/merchant Smart Discovery + Safe Late-Night Matching
- merchant_growth: pricing/discount advice, demand forecasting
- fraud_risk: order risk scoring, driver trust scoring, customer anomaly detection
- planner: multi-dimensional analysis. Route here ONLY for cross-cutting questions
  that span multiple personas, cities, time windows, or segments — e.g.
  "compare X across Y by Z", "strategy memo on...", "deep dive into...".
  The planner spawns ephemeral specialists for each sub-task. Do NOT route here
  for simple single-domain questions; those go to one specialist directly.
- FINISH: the user's request has been fully answered; end the conversation

==== ROUTING RULES ====

**STRICT ROLE GATING (read this first).** The user's role determines which agent can be the
FIRST agent invoked. Cross-routing by intent alone is forbidden — a merchant asking a
customer-flavoured question is NOT served by the customer agent.

  - role='driver'   → driver_success may be invoked first; no other primary agent.
  - role='customer' → customer_convenience may be invoked first (and may start the
                      customer-order chain that calls fraud_risk + driver_success as services).
  - role='merchant' → merchant_growth may be invoked first (and may start the
                      merchant-coverage chain that calls driver_success as a service).
  - role='admin'    → any agent.

If the user's question is clearly outside their role's natural domain, **still route to their
role's primary agent** — it has been trained to politely decline and redirect. Do NOT route
to a different role's agent just because the question fits that domain.

Examples of role-appropriate routing on off-topic questions:
  - User (merchant) asks "what should I eat for lunch?"     → merchant_growth (will decline)
  - User (driver) asks "find me a pricing analysis"         → driver_success (will decline)
  - User (customer) asks "where should I drive tonight?"    → customer_convenience (will decline)

**Single-agent flow (default).** For most in-role questions, pick the role's agent then FINISH:
  - Driver question (where/when/how to earn) → driver_success → FINISH
  - Customer asks "what should I eat?" → customer_convenience → FINISH
  - Merchant asks about pricing/forecast → merchant_growth → FINISH
  - Standalone risk check (any role) → fraud_risk → FINISH

**Cross-agent ORDER-PLACEMENT CHAIN.** When the user is *placing an order* (phrases like
"place an order", "order it", "let's order", "go ahead and order", "I want to order X from Y"),
run this chain:
  1. First call: customer_convenience  (build/confirm the basket)
  2. After customer_convenience finishes: fraud_risk  (score the order before placement)
  3. After fraud_risk finishes (and approved/review-only): driver_success  (assign a driver)
  4. After driver_success finishes: FINISH

Detect "where am I in the chain" by reading message history:
  - No assistant messages yet → start at customer_convenience
  - Last assistant message is from customer_convenience proposing items + the user's
    original ask was an order-placement → next is fraud_risk
  - Last assistant message contains a fraud risk score / "approve" / "review" → next is driver_success
  - Last assistant message contains a matched driver / "Driver Match" → FINISH

**Cross-agent TRUST CHECK CHAIN.** When the user asks "is this driver trustworthy?" or "find me
a safe driver", customer_convenience can handle it solo via find_safe_late_night_drivers — no
chain needed unless they want to actually place an order.

**Cross-agent MERCHANT-DEMAND-PREP CHAIN.** When a *merchant* asks about preparing for demand
or driver coverage (phrases like "will my orders get picked up?", "should I staff up?", "are
there enough drivers Friday night?", "is delivery covered for the dinner rush?"), run:
  1. First call: merchant_growth  (forecast demand and surface the zone + time)
  2. After merchant_growth finishes (its reply will mention "Driver coverage check needed"
     or describe a specific zone + time): route to driver_success  (estimates driver supply)
  3. After driver_success finishes: FINISH

Detect "where am I in the merchant chain" by reading message history:
  - User is a merchant + last message is the original ask → merchant_growth
  - Last assistant message is from merchant_growth and mentions "Driver coverage check needed"
    or otherwise frames a capacity question → driver_success
  - Last assistant message addresses driver supply / availability → FINISH

Hard rules:
- Don't loop the same agent twice in a row unless it explicitly asked for more info.
- If unsure between FINISH and continuing a chain, prefer FINISH.

Examples:
- User (driver): "Where should I drive on Friday night?"     → driver_success
- User (customer): "Suggest something for dinner"             → customer_convenience
- After driver_success replies with recommendations           → FINISH
- User (customer): "Place an order for vegetarian Thai"       → customer_convenience  (chain start)
- ↑ then customer_convenience replies proposing an item       → fraud_risk  (chain step 2)
- ↑ then fraud_risk replies with risk score 18/100, approved  → driver_success  (chain step 3)
- ↑ then driver_success replies with matched driver           → FINISH
- User (merchant): "Will my orders be covered Friday 7pm?"    → merchant_growth  (chain start)
- ↑ then merchant_growth replies with forecast + coverage ask → driver_success  (chain step 2)
- ↑ then driver_success replies with driver availability      → FINISH
"""


# Fallback agent per user role when the supervisor LLM fails to produce
# a valid RouteDecision. For non-admins this routes to their persona's
# natural primary agent. For admins, we infer intent from keywords in
# the most recent user message via _admin_intent_fallback() below.
ROLE_FALLBACK_AGENT = {
    "driver":   "driver_success",
    "customer": "customer_convenience",
    "merchant": "merchant_growth",
    # admin/unknown handled dynamically by _admin_intent_fallback()
}

# Keyword → agent buckets for admin intent inference.
# The Planner bucket goes FIRST so multi-dimensional / analytical queries
# pre-empt the simpler single-domain matches that follow.
_AGENT_KEYWORDS: list[tuple[str, list[str]]] = [
    ("planner", [
        " compare ", "compare ", " vs ", " vs.", " versus ",
        "across cities", "across all", "across segments",
        "strategy memo", "give me a memo", "deep dive",
        "multi-dimensional", "report on", "synthesize",
        "cross-cut", "cross cut", "cross-persona",
    ]),
    ("merchant_growth", [
        "forecast", "demand", "pricing", "discount", "bundle", "menu",
        "competitor", "sales", "revenue", "merchant", "restaurant",
        "weekend order", "prep time", "AOV", "top item",
    ]),
    ("customer_convenience", [
        "eat", "vegetarian", "vegan", "halal", "discover", "hungry",
        "recommend", "food", "cuisine", "thai", "indian", "place an order",
        "what should i order", "order me",
    ]),
    ("fraud_risk", [
        "fraud", "risk score", "block", "suspicious", "anomaly",
        "trust score", "verify", "watchlist", "flagged",
    ]),
    ("driver_success", [
        "plan my day", "where should i drive", "earnings window",
        "incentive", "peak", "hotspot", "zone", "route", "driver",
    ]),
]


def _admin_intent_fallback(messages: list) -> str:
    """For admin / unknown roles, infer the right agent from the latest user
    message's keywords. Falls back to driver_success if nothing matches."""
    last_user = ""
    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage):
            last_user = str(msg.content or "")
            break
    text = last_user.lower()
    if not text:
        return "driver_success"
    for agent_name, kws in _AGENT_KEYWORDS:
        if any(kw in text for kw in kws):
            return agent_name
    return "driver_success"


def _pick_fallback(role: str, messages: list) -> str:
    if role in ROLE_FALLBACK_AGENT:
        return ROLE_FALLBACK_AGENT[role]
    return _admin_intent_fallback(messages)


def supervisor_node(state: AgentState) -> dict:
    """Decide which agent runs next.

    Hardened against non-Anthropic Bedrock models that occasionally return
    None or malformed objects from `with_structured_output()` (the most
    common failure mode for Qwen3 / DeepSeek / etc., since structured-output
    support varies across providers). On any failure we record a
    hallucination and fall back to the user's role-default agent.
    """
    role = state.get("user_role") or "unknown"
    role_note = SystemMessage(content=f"The current user's role is: {role}.")
    fallback = _pick_fallback(role, list(state.get("messages", [])))

    llm = get_haiku(model_override=state.get("model_override"), temperature=0).with_structured_output(RouteDecision)

    _t0 = time.time()
    decision: RouteDecision | None = None
    failure_reason: str | None = None
    try:
        decision = llm.invoke(
            [SystemMessage(content=SUPERVISOR_PROMPT), role_note] + list(state.get("messages", []))
        )
    except Exception as e:
        failure_reason = f"invoke_exception:{type(e).__name__}"
    _duration_ms = int((time.time() - _t0) * 1000)

    # --- Validate ----
    _allowed = {"driver_success", "customer_convenience", "merchant_growth", "fraud_risk", "planner", "FINISH"}
    _reasons: list[str] = []
    next_agent: str

    if decision is None or not hasattr(decision, "next_agent") or decision.next_agent is None:
        # Model returned nothing parseable — fall back to role default.
        _reasons.append(failure_reason or "structured_output_null")
        _reasons.append(f"fallback_to:{fallback}")
        next_agent = fallback
        reason_text = f"structured-output failure, defaulted to {fallback}"
    elif decision.next_agent not in _allowed:
        # Model returned an invalid value — flag + fall back.
        _reasons.append(f"invalid_agent:{decision.next_agent}")
        _reasons.append(f"fallback_to:{fallback}")
        next_agent = fallback
        reason_text = f"invalid route {decision.next_agent!r}, defaulted to {fallback}"
    else:
        next_agent = decision.next_agent
        reason_text = decision.reason

    # ---- Telemetry ----
    track_agent_call(
        agent_name="supervisor", state=state, ai_message=None,
        duration_ms=_duration_ms, hallucinated_reasons=_reasons or None,
    )

    # We append a small AIMessage so the trace shows the routing choice.
    # Wrap it as an internal note (won't be shown to the end-user directly).
    note = AIMessage(content=f"[supervisor] → {next_agent}: {reason_text}")
    return {"next_agent": next_agent, "messages": [note]}


# ------------------------------ graph wiring ----------------------------------
# Whitelist of legal next-agent values. Any hallucination (common with
# non-Anthropic Bedrock models that don't reliably honor structured output)
# is clamped to END so the chat doesn't blow up with KeyError.
_KNOWN_AGENTS = {
    "driver_success",
    "customer_convenience",
    "merchant_growth",
    "fraud_risk",
    "planner",
}


def _route_after_supervisor(state: AgentState) -> str:
    nxt = state.get("next_agent")
    if nxt is None or nxt == "FINISH" or nxt not in _KNOWN_AGENTS:
        return END
    return nxt  # type: ignore[return-value]


def build_supervisor_graph():
    g = StateGraph(AgentState)

    g.add_node("supervisor", supervisor_node)
    g.add_node("driver_success", driver_success.node)
    g.add_node("customer_convenience", customer_convenience.node)
    g.add_node("merchant_growth", merchant_growth.node)
    g.add_node("fraud_risk", fraud_risk.node)
    g.add_node("planner", planner.node)

    g.set_entry_point("supervisor")
    g.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "driver_success": "driver_success",
            "customer_convenience": "customer_convenience",
            "merchant_growth": "merchant_growth",
            "fraud_risk": "fraud_risk",
            "planner": "planner",
            END: END,
        },
    )
    # Every specialist returns to the supervisor, which will pick FINISH next.
    for agent in ["driver_success", "customer_convenience", "merchant_growth", "fraud_risk", "planner"]:
        g.add_edge(agent, "supervisor")

    return g.compile()


# Singleton — built once on import.
GRAPH = build_supervisor_graph()
