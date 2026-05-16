# GrabWise

**Multi-agent AI for Grab's ecosystem.** A Supervisor routes user requests across four specialist agents — Driver Success, Customer Convenience, Merchant Growth, and Fraud & Risk — each with its own tools and deck-aligned pillars. Two cross-agent flows are wired end-to-end.

> See [GRABWISE_BUILD_PLAN.md](GRABWISE_BUILD_PLAN.md) for the full hackathon plan.

---

## Architecture at a glance

```
                        ┌──────────────────────────┐
                        │      Supervisor          │
                        │  (Haiku, structured      │
                        │  routing decisions)      │
                        └──────────┬───────────────┘
                                   │
       ┌───────────────┬───────────┼───────────────┬─────────────────┐
       ▼               ▼           ▼               ▼                 ▼
┌─────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  FINISH
│ Driver      │ │ Customer     │ │ Merchant     │ │ Fraud &      │
│ Success     │ │ Convenience  │ │ Growth       │ │ Risk         │
│ (9 tools)   │ │ (5 tools)    │ │ (6 tools)    │ │ (4 tools)    │
└─────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

**Cross-agent flows wired today:**

1. **Customer order placement** → Customer ▸ Fraud ▸ Driver ▸ FINISH
2. **Merchant demand prep** → Merchant ▸ Driver ▸ FINISH

---

## What's working today

- ✅ Supervisor (LangGraph, Haiku-routed) with structured output
- ✅ All four agents are real, ReAct-loop, tool-bound
- ✅ **24 tools** across the four agents, each mapped to a deck pillar
- ✅ Two cross-agent chains with explicit routing rules
- ✅ Synthetic dataset: 5 cities, 200 drivers, 50 merchants, 50 customers, 3000 orders, ~400 menu items, 10 incentives
- ✅ LLM factory: AWS Bedrock when configured, Anthropic direct API as a fallback
- ✅ FastAPI + SSE backend streaming agent activity in real time
- ✅ Tailwind UI with persona switcher + live Agent Activity panel + chain-aware multi-bubble chat
- ✅ Rich-flavored CLI smoke test for fast iteration

---

## Quickstart

Assumes Python 3.13 (see [Setup](#setup) for venv details).

```bash
source "/Users/anushaarra/Downloads/Grab-Agent/.venv/bin/activate"
cd "/Users/anushaarra/Downloads/Grab-Agent/Grab - Hackathon"

pip install -r requirements.txt        # one-time
cp .env.example .env                   # fill in ANTHROPIC_API_KEY (or AWS_*)
python -m backend.db.seed              # creates grabwise.db
uvicorn backend.main:app --reload --port 8000
```

Then open **http://localhost:8000**.

---

## Setup (first time)

The project lives in `Grab-Agent/Grab - Hackathon/`. The virtualenv is one level up at `Grab-Agent/.venv/`.

### 1. Activate the venv

```bash
source "/Users/anushaarra/Downloads/Grab-Agent/.venv/bin/activate"
cd "/Users/anushaarra/Downloads/Grab-Agent/Grab - Hackathon"
```

If you don't have the venv yet:

```bash
python3.13 -m venv "/Users/anushaarra/Downloads/Grab-Agent/.venv"
source "/Users/anushaarra/Downloads/Grab-Agent/.venv/bin/activate"
```

> **Python 3.13 is recommended.** Python 3.14 also works with the loose pins in `requirements.txt`, but several optional ML libs you may add later (chromadb, faiss) lag on 3.14 wheels.

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Open `.env` and set **one** of:

- **AWS Bedrock (target stack at the hackathon)** — uncomment and fill `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`. Verify access in the Bedrock console for the Claude Sonnet + Haiku model IDs in `.env.example`.
- **Anthropic direct (for local dev before hackathon credits arrive)** — set `ANTHROPIC_API_KEY` from console.anthropic.com (note: separate from a Claude Pro subscription; you need API credits in the console).

The LLM factory in `backend/llm/bedrock.py` auto-picks Bedrock if AWS creds are present, otherwise Anthropic.

Quick check:

```bash
python -c "from backend.config import settings; \
  print('provider:', 'bedrock' if settings.use_bedrock else 'anthropic-direct'); \
  print('key set:', bool(settings.anthropic_api_key) or settings.use_bedrock)"
```

### 4. Seed the database

```bash
python -m backend.db.seed
```

This creates `grabwise.db` (SQLite) in the project root and populates it with all synthetic data. Run again any time to reset to a clean known state.

---

## Run

### Web UI (recommended)

```bash
uvicorn backend.main:app --reload --port 8000
```

Open **http://localhost:8000**. The page has:

- A **persona switcher** in the header (🚲 Driver · 🍜 Customer · 🏪 Merchant) with a per-role user dropdown.
- A **chat panel** on the left with role-specific suggested prompts.
- An **Agent Activity panel** on the right that animates as supervisor routing decisions and tool calls stream in via SSE. Color-coded by agent: purple Supervisor, green Driver, blue Customer, amber Merchant, rose Fraud.
- A **provider badge** showing which LLM backend is active (`bedrock` or `anthropic-direct`).

Switching personas or users clears the chat — each persona/user is its own clean session.

### Stop / restart

`Ctrl+C` in the uvicorn terminal. If the port is held by a stuck process:

```bash
lsof -ti:8000 | xargs kill -9
```

### CLI smoke test

For fast iteration without the UI:

```bash
# Single-shot
python -m scripts.run_chat --role driver --user-id 1 \
  --message "It's Friday evening — when's my peak earning window?"

# Interactive REPL
python -m scripts.run_chat --role driver --user-id 1
```

The CLI prints the supervisor's routing decisions, every tool call's input/output, and the final answer with Rich formatting.

---

## Demo prompts

### Driver — full tools

Each prompt below exercises a different deck pillar.

```
"When's my peak earning window on Friday nights?"
"Where in my city will demand be highest at 7pm Saturday?"
"How can I optimize my earnings? Where am I losing money?"
"It's Friday afternoon — give me a complete plan to maximize tonight."
```

Look for **Peak Window**, **Geo Hotspot**, and **Earning Optimization** labels in the agent's reply.

### Customer — full tools

```
"I'm vegetarian and hungry — what should I order tonight?"
"What have I been ordering lately? Any new spots I should try?"
"It's late — find me a safe driver if I order around midnight."
"Show me halal Indian places with prep time under 15 minutes."
```

The first three exercise **Smart Discovery**; the third also fires **Safe Late-Night Matching**.

### Merchant — full tools

```
"Suggest pricing or bundle actions for my menu."
"What demand should I expect this Friday evening?"
"My weekend orders are dropping. What should I do?"
```

Returns concrete pricing/discount/bundle suggestions citing real menu items + cuisine-median benchmarks.

### Cross-agent flows (the wow shots)

**Customer-order chain** — three agents in sequence (Customer ▸ Fraud ▸ Driver):

```
"Place an order for vegetarian Thai — check it's safe and assign a driver."
"Order me a halal dinner — make sure it's safe at this hour."
```

**Merchant-coverage chain** — two agents in sequence (Merchant ▸ Driver):

```
"Prepping for the dinner rush — forecast demand and check driver coverage."
"Will my orders get picked up Friday at 7pm?"
```

In both, the chat shows a separate labeled bubble per agent and the Activity panel lights up with each agent's tool calls.

---

## Project structure

```
.
├── backend/
│   ├── main.py                       # FastAPI app + static mount
│   ├── config.py                     # pydantic-settings, reads .env
│   ├── state.py                      # LangGraph AgentState
│   ├── api/
│   │   ├── chat.py                   # POST /api/chat — multi-event SSE
│   │   ├── users.py                  # GET /api/users?role=...
│   │   └── schemas.py                # Pydantic request/response models
│   ├── agents/
│   │   ├── supervisor.py             # supervisor graph + chain rules
│   │   ├── driver_success.py         # ReAct (9 tools)
│   │   ├── customer_convenience.py   # ReAct (5 tools)
│   │   ├── merchant_growth.py        # ReAct (6 tools)
│   │   └── fraud_risk.py             # ReAct (4 tools)
│   ├── tools/
│   │   ├── driver_tools.py           # incl. match_driver_for_order, estimate_driver_availability
│   │   ├── customer_tools.py         # search_merchants, find_safe_late_night_drivers, ...
│   │   ├── merchant_tools.py         # forecast_merchant_demand, suggest_pricing_actions, ...
│   │   └── fraud_tools.py            # score_order_risk, score_driver_trust, ...
│   ├── llm/bedrock.py                # LLM factory: Bedrock or Anthropic
│   └── db/
│       ├── database.py
│       ├── models.py                 # SQLAlchemy ORM (City/Driver/Merchant/MenuItem/Customer/Order/Incentive)
│       └── seed.py                   # Faker-driven synthetic data + cuisine menu pools
├── static/
│   └── index.html                    # single-page UI (Tailwind via CDN, vanilla JS, SSE consumer)
├── scripts/
│   └── run_chat.py                   # CLI smoke test (Rich-flavored)
├── requirements.txt
├── .env.example
├── GRABWISE_BUILD_PLAN.md
├── grabwise.db                       # generated by seed.py — gitignored
└── README.md
```

---

## Agents and pillars

Each agent's tools map directly to a named pillar from the pitch deck.

### Driver Success Agent
| Pillar | Tool |
|---|---|
| Peak Earning Window | `get_peak_earning_windows` |
| Geo Hotspots | `predict_demand_hotspots`, `get_busy_zones` |
| Earning Optimization | `get_savings_recommendations`, `get_driver_earnings` |
| (cross) Order matching | `match_driver_for_order` |
| (cross) Capacity for merchants | `estimate_driver_availability` |
| Supporting | `get_driver_profile`, `get_active_incentives` |

### Customer Convenience Agent
| Pillar | Tool |
|---|---|
| Smart Discovery | `search_merchants`, `get_merchant_menu`, `get_customer_recent_orders` |
| Safe Late-Night Matching | `find_safe_late_night_drivers` |
| Supporting | `get_customer_profile` |

### Merchant Growth Agent
| Pillar | Tool |
|---|---|
| AI Pricing & Discount Suggestions | `suggest_pricing_actions`, `get_top_items`, `get_competitor_signals` |
| Demand Forecasting & Trend Insights | `forecast_merchant_demand`, `get_merchant_order_rollup` |
| Supporting | `get_merchant_profile` |

### Fraud & Risk Agent
| Pillar | Tool |
|---|---|
| Driver Trust Scoring | `score_driver_trust` |
| Trusted Late-Night Matching | (uses `score_driver_trust` + late-night flag) |
| Customer Anomaly Detection | `score_customer_anomaly` |
| Order risk (chain step) | `score_order_risk` |
| Supporting | `get_transaction_signals` |

---

## What's next

- Multi-turn memory via `MemorySaver` checkpointer + `thread_id` so conversations persist
- Token-level streaming so replies feel snappier
- Chroma RAG over `menu_items` for semantic dietary/preference search
- Bedrock Guardrails on the Fraud agent for free PII redaction
- A proactive driver alert mode (CLI/scheduled) — covers the deck's "Alerts drivers" promise

When AWS Bedrock credits arrive at the hackathon, no code changes are needed — just fill the `AWS_*` block in `.env` and the LLM factory switches automatically.

---

## Troubleshooting

- **`ModuleNotFoundError: No module named 'backend'`** — you're not in the project root. `cd` into `Grab - Hackathon/` first.
- **Address already in use on port 8000** — `lsof -ti:8000 | xargs kill -9`, then restart.
- **`anthropic.BadRequestError: Your credit balance is too low`** — you have a Claude Pro subscription, not API credits. Add credits at console.anthropic.com → Plans & Billing.
- **UI shows "no users"** for a persona — re-seed: `python -m backend.db.seed` (this is required after the menu_items model was added).
- **Chain doesn't fire end-to-end** (e.g. Customer answers but Fraud isn't called) — paste the Agent Activity panel output back to the AI assistant; it's almost always a supervisor-prompt tuning issue.
