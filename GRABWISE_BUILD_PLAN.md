# GrabWise — Coding Strategy & Build Plan

A pragmatic, hackathon-shaped plan for building GrabWise: a multi-agent AI system with a Supervisor coordinating Driver Success, Customer Convenience, Merchant Growth, and Fraud & Risk agents. Targets a 36-hour build.

---

## 1. Strategy: how to actually finish this

**Build vertically, not horizontally.** Get one full slice working end-to-end (synthetic data → DB → one agent → supervisor → API → UI) before adding the other three agents. Most hackathon teams die because they build all four agents in parallel and never integrate. Resist that.

**Two agents deep, two agents shallow.** Driver Success and Customer Convenience are the most demoable — build them fully with tools and RAG. Merchant Growth and Fraud & Risk get simpler implementations (fewer tools, more LLM-prompt-driven). The supervisor still routes to all four, so the orchestration story stays intact.

**Fake the ML, fake the scale.** "Demand forecasting," "trust-aware matching," "personalization" are all LLM-reasoning-over-synthetic-data in this prototype. Don't train models. Don't worry about throughput. The demo runs on ~5000 synthetic rows.

**Demo-driven development.** Write the demo script before the code. Every feature must serve one of the demo scenarios in §11. If it doesn't, cut it.

**One person owns the supervisor + LangGraph wiring.** This is the spine. If two people touch it, it breaks. Other contributors build agents and tools that plug into a stable interface.

---

## 2. Final tech stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI + uvicorn | Async, SSE-friendly, fast to scaffold |
| Orchestration | LangGraph | Supervisor pattern is first-class |
| LLM | AWS Bedrock — Claude Sonnet (agents) + Claude Haiku (supervisor routing) | Hackathon credits + speed/cost split |
| LLM SDK | `langchain-aws` `ChatBedrockConverse` | Modern Converse API, tool calling works cleanly |
| Embeddings | Bedrock Titan Text Embeddings v2 | No extra API keys, stays in AWS |
| Vector DB | ChromaDB (persistent local) | Lightweight, no service to run |
| Relational DB | PostgreSQL 16 via SQLAlchemy 2.0 + asyncpg | Real-feeling for demo; SQLite if solo |
| Synthetic data | Faker + custom generators | Drivers, merchants, customers, orders |
| Frontend | Next.js 14 (App Router) + Tailwind + shadcn/ui | Polished, supports streaming UI |
| Streaming | Server-Sent Events (SSE) | Simpler than WebSockets for one-way agent updates |
| Local infra | Docker Compose (Postgres + Chroma + API) | One-command bring-up |
| Deployment (optional) | Railway or Render | Public URL for judges if needed |

**Skipped on purpose:** LangChain (use `langchain_core` only for messages/tools), LlamaIndex (Chroma's own retrieval is enough), Redis, Celery, Kubernetes, Bedrock Agents (we're using LangGraph instead).

---

## 3. Repo structure

```
grabwise/
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── README.md
├── Makefile                       # `make seed`, `make run`, `make demo`
├── backend/
│   ├── main.py                    # FastAPI entrypoint
│   ├── config.py                  # pydantic-settings, reads .env
│   ├── state.py                   # LangGraph AgentState TypedDict
│   ├── db/
│   │   ├── postgres.py            # async engine + session
│   │   ├── models.py              # SQLAlchemy ORM
│   │   └── seed.py                # Faker-based data generator
│   ├── vectorstore/
│   │   ├── chroma.py              # collection setup
│   │   ├── embeddings.py          # Bedrock Titan wrapper
│   │   └── ingest.py              # one-shot embedding script
│   ├── llm/
│   │   └── bedrock.py             # Sonnet + Haiku factory
│   ├── tools/
│   │   ├── driver_tools.py
│   │   ├── customer_tools.py
│   │   ├── merchant_tools.py
│   │   └── fraud_tools.py
│   ├── agents/
│   │   ├── supervisor.py          # build_supervisor_graph()
│   │   ├── driver_success.py
│   │   ├── customer_convenience.py
│   │   ├── merchant_growth.py
│   │   └── fraud_risk.py
│   └── api/
│       ├── chat.py                # POST /chat (SSE)
│       ├── health.py
│       └── schemas.py
└── frontend/
    ├── package.json
    ├── tailwind.config.ts
    ├── app/
    │   ├── layout.tsx
    │   ├── page.tsx
    │   └── api/chat/route.ts      # proxies SSE from backend
    └── components/
        ├── ChatPanel.tsx
        ├── AgentActivityPanel.tsx
        ├── MessageBubble.tsx
        └── PersonaSwitcher.tsx    # toggle driver/customer/merchant view
```

---

## 4. Synthetic data model

Keep it small and internally consistent. Generate once, commit the SQL dump so teammates start from the same world.

**Tables (Postgres):**
- `cities` (5 rows: Singapore, Jakarta, Bangkok, Manila, KL)
- `customers` (50 rows — name, city_id, dietary_prefs, signup_date)
- `drivers` (200 rows — name, city_id, vehicle_type, rating, joined_date)
- `merchants` (100 rows — name, city_id, cuisine, rating, avg_prep_min)
- `menu_items` (~10 per merchant — name, description, price, tags)
- `orders` (5000 rows over last 90 days — customer_id, merchant_id, driver_id, total, status, created_at, dropoff_zone)
- `order_items` (line items)
- `transactions` (matching orders + a few standalone wallet txns)
- `driver_earnings_daily` (rollup view, computed from orders)
- `incentives` (5–10 active campaigns — bonus zones, completion bonuses)

**Vector collections (Chroma):**
- `merchant_menus` — each menu item embedded with merchant name + tags
- `driver_feedback` — synthetic short feedback notes per driver
- `customer_history_summaries` — one-sentence summary per customer's recent ordering pattern

Use Faker for names, but **realistic distributions matter**: weight orders to weekends and lunch/dinner peaks, give some drivers high ratings and some low, cluster orders by zone. Demos look fake when data looks uniform.

---

## 5. Agent design — what each one actually does

Define each agent as: **prompt + tools + 1-line output contract**. Anything else is overengineering for a hackathon.

### Driver Success Agent (build deeply)
**Goal:** Help a driver earn more.
**Tools:**
- `get_driver_profile(driver_id)` — rating, city, vehicle, tenure
- `get_driver_earnings(driver_id, days)` — daily earnings rollup
- `get_busy_zones(city_id, day_of_week, hour)` — top 5 zones by order volume
- `get_active_incentives(city_id, vehicle_type)` — bonus opportunities
**Prompt:** acts like an earnings coach, makes 2-3 concrete suggestions with reasoning.

### Customer Convenience Agent (build deeply)
**Goal:** Smart basket / personalized order suggestion.
**Tools:**
- `get_customer_profile(customer_id)` — dietary prefs, city
- `get_customer_recent_orders(customer_id, n)`
- `search_merchants(city_id, query, dietary_filter)` — uses Chroma over `merchant_menus`
- `get_merchant_menu(merchant_id)`
**Prompt:** suggests one merchant + a 2-3 item basket with personalization rationale.

### Merchant Growth Agent (build shallow)
**Goal:** Help a merchant decide what to add/promote.
**Tools:**
- `get_merchant_orders(merchant_id, days)` — daily volume
- `get_top_items(merchant_id, days)`
- `get_competitor_signals(city_id, cuisine)` — same-cuisine merchants' top items via Chroma
**Prompt:** outputs one growth recommendation with supporting numbers.

### Fraud & Risk Agent (build shallow)
**Goal:** Score a transaction or driver-customer match.
**Tools:**
- `get_transaction(txn_id)`
- `get_customer_history_signals(customer_id)` — refunds, disputes count
- `get_driver_risk_signals(driver_id)` — recent rating drops, cancel rate
**Prompt:** returns a risk score 0-100 with two reasons. **Wrap with Bedrock Guardrails** for PII redaction — free credibility.

### Supervisor (built last, but designed first)
- Uses **Haiku** for routing speed.
- System prompt lists all four agents with one-line capability blurbs.
- Output: structured `{next_agent: "driver_success" | ..., reason: str}` or `{next_agent: "FINISH"}`.
- For cross-agent flows: returns to supervisor after each agent runs; loop until FINISH.

---

## 6. LangGraph state schema

Single shared state. Keep it small.

```python
# backend/state.py
from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_role: Literal["customer", "driver", "merchant", "admin"]
    user_id: str
    next_agent: str | None        # set by supervisor, read by router
    agent_trace: list[dict]       # for the activity panel: [{agent, tool, input, output, ts}]
```

`agent_trace` is what the frontend's Agent Activity Panel reads — every tool call appends an entry.

---

## 7. Supervisor graph wiring (reference snippet)

```python
# backend/agents/supervisor.py
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from backend.state import AgentState
from backend.llm.bedrock import get_haiku, get_sonnet
from backend.agents import driver_success, customer_convenience, merchant_growth, fraud_risk

SUPERVISOR_PROMPT = """You route requests to the right agent. Available agents:
- driver_success: earnings guidance, zone/time recommendations, incentive lookup
- customer_convenience: meal/merchant suggestions, smart basket, dietary-aware
- merchant_growth: menu/promotion advice, demand insights for merchants
- fraud_risk: transaction risk scoring, trust signals
- FINISH: when the user's request is fully answered

Return JSON: {"next_agent": "<name>", "reason": "<one sentence>"}"""

def supervisor_node(state: AgentState) -> dict:
    llm = get_haiku().with_structured_output(RouteDecision)
    decision = llm.invoke([("system", SUPERVISOR_PROMPT)] + state["messages"])
    return {"next_agent": decision.next_agent}

def route(state: AgentState) -> str:
    return END if state["next_agent"] == "FINISH" else state["next_agent"]

def build_supervisor_graph():
    g = StateGraph(AgentState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("driver_success", driver_success.node)
    g.add_node("customer_convenience", customer_convenience.node)
    g.add_node("merchant_growth", merchant_growth.node)
    g.add_node("fraud_risk", fraud_risk.node)
    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor", route)
    for a in ["driver_success", "customer_convenience", "merchant_growth", "fraud_risk"]:
        g.add_edge(a, "supervisor")  # always return to supervisor
    return g.compile()
```

Each agent node is a small ReAct loop: bind tools to Sonnet, invoke, append to `agent_trace`, return.

---

## 8. Bedrock setup (DO THIS FIRST — hour zero)

1. Log into the Bedrock console in the region you've been assigned.
2. Under **Model access**, request: Claude Sonnet (latest), Claude Haiku, Titan Text Embeddings v2. Wait for "Access granted."
3. Verify with a quick boto3 test:

```python
import boto3, json
client = boto3.client("bedrock-runtime", region_name="us-east-1")
resp = client.converse(
    modelId="anthropic.claude-sonnet-4-20250514-v1:0",  # check exact ID at hackathon time
    messages=[{"role": "user", "content": [{"text": "Say hello"}]}],
)
print(resp["output"]["message"]["content"][0]["text"])
```

4. Pin versions in `requirements.txt`:
```
fastapi==0.115.*
uvicorn[standard]==0.32.*
sqlalchemy==2.0.*
asyncpg==0.30.*
pydantic-settings==2.6.*
boto3==1.35.*
langchain-aws==0.2.*
langchain-core==0.3.*
langgraph==0.2.*
chromadb==0.5.*
faker==30.*
sse-starlette==2.1.*
```

Pin them now or the LangChain + Bedrock combo will surprise you mid-hackathon.

---

## 9. Phase-by-phase plan

### Phase 0 — Hour 0 to 1: Bedrock + repo skeleton
- [ ] Confirm Bedrock model access (above)
- [ ] `git init`, push to GitHub, set up branch protection
- [ ] Scaffold repo structure (§3)
- [ ] `.env.example` with `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `DATABASE_URL`, `CHROMA_PATH`
- [ ] `docker-compose.yml` with postgres + a chroma volume mount + the api service
- [ ] FastAPI `main.py` with `/health` returning 200

**Done = `docker compose up` works, `curl localhost:8000/health` returns ok.**

### Phase 1 — Hour 1 to 5: Data layer
- [ ] `db/models.py` — all tables from §4
- [ ] Run migrations (use `alembic` or just `Base.metadata.create_all` for hackathon speed)
- [ ] `db/seed.py` — Faker-driven generator with realistic distributions
- [ ] `make seed` populates Postgres in <30s
- [ ] `vectorstore/embeddings.py` — Titan wrapper (Chroma `EmbeddingFunction`)
- [ ] `vectorstore/ingest.py` — embeds menu items, driver feedback, customer summaries into Chroma
- [ ] Sanity query: `chroma_client.query("vegetarian thai")` returns sensible results

**Done = data exists, embeddings exist, you can query both.**

### Phase 2 — Hour 5 to 9: First agent end-to-end
- [ ] `llm/bedrock.py` — `get_sonnet()` and `get_haiku()` factories returning `ChatBedrockConverse`
- [ ] `tools/driver_tools.py` — all 4 driver tools, decorated with `@tool`
- [ ] `agents/driver_success.py` — ReAct-style node binding tools to Sonnet
- [ ] Build a **temporary** single-agent graph (no supervisor yet) and test from a Python REPL
- [ ] `api/chat.py` — `POST /chat` accepting `{user_id, role, message}`, returns SSE stream
- [ ] SSE event types: `message_chunk`, `tool_call`, `tool_result`, `done`
- [ ] Manual test: hit `/chat` with "How can I earn more this Friday?" and watch tools fire

**Done = one agent works end-to-end via API, you can see tool calls happening.**

### Phase 3 — Hour 9 to 14: Supervisor + remaining agents
- [ ] `agents/supervisor.py` from §7
- [ ] Implement remaining three agents (Customer deeply, Merchant + Fraud shallowly)
- [ ] `tools/customer_tools.py`, `merchant_tools.py`, `fraud_tools.py`
- [ ] Wire all into the supervisor graph
- [ ] Test routing: drive 4 different prompts, confirm correct agent picked each time
- [ ] Implement one **cross-agent flow**: customer places mock order → supervisor calls Customer agent → then routes to Fraud agent → returns combined response

**Done = supervisor routes correctly across all 4 agents, one cross-agent flow works.**

### Phase 4 — Hour 14 to 17: RAG and personalization
- [ ] Customer agent's `search_merchants` queries Chroma `merchant_menus` with metadata filtering by city + dietary tag
- [ ] Merchant agent's `get_competitor_signals` queries Chroma for similar-cuisine merchants' top items
- [ ] Add **one** persisted personalization fact: store `customer_history_summary` in Chroma after each conversation. Customer agent retrieves it on the next turn. This is your "semantic memory" demo.
- [ ] Add Bedrock Guardrails to the Fraud agent's output (PII filter)

**Done = RAG visible in two agents, one persistent memory loop, guardrail wired.**

### Phase 5 — Hour 17 to 26: Frontend
- [ ] `npx create-next-app frontend --tailwind --app --typescript`
- [ ] Install shadcn: `card`, `button`, `input`, `scroll-area`, `badge`, `tabs`
- [ ] `PersonaSwitcher` — pick a fake user (driver/customer/merchant) from a dropdown of seeded users
- [ ] `ChatPanel` — text input + message list, calls `/api/chat/route.ts` which proxies SSE from FastAPI
- [ ] `AgentActivityPanel` — right-side panel that animates as `tool_call` / `tool_result` events arrive. Show: agent name, tool name, args, brief result. **This is the demo's centerpiece.**
- [ ] `MessageBubble` — renders streaming text with a typing indicator
- [ ] Final touches: GrabWise logo, dark mode, smooth scroll on new messages

**Done = polished UI showing live agent orchestration.**

### Phase 6 — Hour 26 to 32: Demo scenarios + polish
- [ ] Hardcode `make demo` that resets the DB to a known state
- [ ] Pre-write 3-4 demo prompts (§11) and rehearse — make sure each takes <30s end-to-end
- [ ] Build a **fallback canned response** path: env flag `DEMO_MODE=true` returns deterministic responses if Bedrock latency or rate-limit becomes a problem live
- [ ] Capture a screen recording of a perfect run as backup
- [ ] Logo, landing copy, "How it works" diagram in the README

### Phase 7 — Hour 32 to 36: Pitch + buffer
- [ ] Slides updated with screenshots of the working app (replace the architecture-only deck)
- [ ] Practice pitch twice with the demo
- [ ] Reserve final 2 hours for fixes — something will break

---

## 10. Hard rules to not lose time

1. **No new dependencies after Phase 3.** Anything you don't have by hour 9 isn't going in.
2. **No real auth.** A dropdown selecting a fake user_id is enough.
3. **No payments integration.** "Pay" is a button that mutates an order row and triggers the Fraud agent.
4. **No real maps.** Zones are strings ("Bukit Timah", "Orchard"). Don't pull a map library.
5. **Don't touch the supervisor prompt after Phase 3 unless routing is provably broken.** Tweaking it cascades into every agent test.
6. **Commit every 30 minutes.** A working `main` is more valuable than a clever local change.
7. **One person on infra/integration, others on agents.** Don't let everyone touch `docker-compose.yml`.

---

## 11. Demo script — the four scenarios

Pre-rehearse exactly these. The judge sees three personas and one cross-agent flow.

**Scenario 1 — Driver (uses Driver Success Agent):**
Persona: driver Aman, Singapore, 4.7 rating.
Prompt: *"It's Friday afternoon — where should I head to maximize earnings tonight?"*
Demo points: tool calls visible (busy zones, active incentives), specific recommendation with rationale.

**Scenario 2 — Customer (uses Customer Convenience Agent + RAG + memory):**
Persona: customer Priya, vegetarian, has ordered Thai twice this month.
Prompt: *"I'm hungry, surprise me with dinner."*
Demo points: agent recalls dietary pref + history (from Chroma summary), retrieves matching merchants, suggests a 3-item basket with reasoning. Run it twice — second time it remembers.

**Scenario 3 — Merchant (uses Merchant Growth Agent):**
Persona: merchant "Bangkok Bites," declining weekend orders.
Prompt: *"Why are my weekend orders dropping and what should I do?"*
Demo points: tool calls show order trend + competitor top items, one concrete recommendation.

**Scenario 4 — Cross-agent flow (the wow moment):**
Customer places an order through the chat. Supervisor:
1. Routes to Customer Agent (basket built)
2. Routes to Fraud Agent (scores 12/100 — low risk)
3. Synthesizes: "Order placed, low risk, ETA 28 min."
The Agent Activity Panel shows three agents lighting up in sequence. **This is what wins.**

---

## 12. Risk register + mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Bedrock model access not granted in time | Med | Hour 0 priority; have OpenAI key as backup wrapped behind same interface |
| Bedrock rate limit during demo | Med | Sequential agent calls (not parallel); `tenacity` retries; `DEMO_MODE` canned fallbacks |
| Supervisor mis-routes | High | Few-shot examples in supervisor prompt; structured output via Pydantic |
| LangGraph/langchain-aws version mismatch | High | Pin everything in Phase 0, no upgrades |
| Frontend SSE chunk parsing breaks | Med | Use `eventsource-parser` on client; test with real Bedrock chunks early, not mocks |
| Chroma persistence weirdness in Docker | Low-Med | Mount a named volume, not bind mount; reset script in Makefile |
| Demo Wi-Fi failure | Med | Run everything localhost; record backup video |
| Synthetic data looks fake | Med | Skewed distributions (peak hours, zone clusters); name merchants with real-sounding names |

---

## 13. Done-criteria checklist (use during final hour)

- [ ] `docker compose up` brings up everything cleanly on a fresh clone
- [ ] `make seed` populates DB + Chroma in <60s
- [ ] All 4 demo scenarios run successfully twice in a row
- [ ] Agent Activity Panel renders for every scenario
- [ ] Cross-agent scenario shows ≥2 agents firing
- [ ] README has setup steps + architecture diagram + demo GIF
- [ ] Slides reference the live product (screenshots, not just architecture)
- [ ] Backup recording exists
- [ ] Repo is public (or judge access set up)

---

## 14. Stretch goals (only if ahead of schedule)

- Voice input (Whisper API for STT, browser TTS for output) — judges love this
- Tracing dashboard with LangSmith — instant credibility on observability
- Real WebSocket-based driver dashboard with simulated location pings
- A second demo language (Bahasa Indonesia) — Bedrock Claude handles it natively, just change the prompt
- Multi-turn memory across sessions persisted in Postgres + Chroma

Treat all of these as *nice-to-have*. Phase 7 buffer time is for fixing what's broken, not adding features.

---

*Last updated: hackathon kickoff. Update this doc as you make scope cuts — they're inevitable and worth tracking.*
