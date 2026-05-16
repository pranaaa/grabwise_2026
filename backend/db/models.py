"""ORM models for GrabWise. Driver-focused for now; customer/merchant tables stay slim."""
from __future__ import annotations
from datetime import datetime
from sqlalchemy import (
    String, Integer, Float, ForeignKey, DateTime, Text, JSON, Boolean
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.database import Base


class City(Base):
    __tablename__ = "cities"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True)
    country: Mapped[str] = mapped_column(String(80))
    # Zones are short labels like "Bukit Timah". Stored as JSON list for simplicity.
    zones: Mapped[list[str]] = mapped_column(JSON, default=list)


class Driver(Base):
    __tablename__ = "drivers"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    phone: Mapped[str] = mapped_column(String(40))
    city_id: Mapped[int] = mapped_column(ForeignKey("cities.id"), index=True)
    vehicle_type: Mapped[str] = mapped_column(String(20))   # bike | car
    rating: Mapped[float] = mapped_column(Float)            # 0.0 - 5.0
    cancel_rate: Mapped[float] = mapped_column(Float)       # 0.0 - 1.0
    joined_date: Mapped[datetime] = mapped_column(DateTime)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    behavior_persona: Mapped[str | None] = mapped_column(String(60), nullable=True)

    # Where the driver typically starts the day. Used as the planner's start node
    # for daily-plan generation. Defaults to first zone of the driver's city on register.
    home_zone: Mapped[str | None] = mapped_column(String(80), nullable=True)

    city: Mapped["City"] = relationship()
    orders: Mapped[list["Order"]] = relationship(back_populates="driver")


class Merchant(Base):
    __tablename__ = "merchants"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(160))
    city_id: Mapped[int] = mapped_column(ForeignKey("cities.id"), index=True)
    cuisine: Mapped[str] = mapped_column(String(40))
    rating: Mapped[float] = mapped_column(Float)
    avg_prep_min: Mapped[int] = mapped_column(Integer)
    zone: Mapped[str] = mapped_column(String(80))
    behavior_persona: Mapped[str | None] = mapped_column(String(60), nullable=True)

    city: Mapped["City"] = relationship()
    menu_items: Mapped[list["MenuItem"]] = relationship(back_populates="merchant", cascade="all, delete-orphan")


class MenuItem(Base):
    """Individual items on a merchant's menu — drives Smart Discovery."""
    __tablename__ = "menu_items"
    id: Mapped[int] = mapped_column(primary_key=True)
    merchant_id: Mapped[int] = mapped_column(ForeignKey("merchants.id"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str] = mapped_column(Text, default="")
    price: Mapped[float] = mapped_column(Float)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)  # vegetarian / vegan / halal / gluten-free
    popularity: Mapped[int] = mapped_column(Integer, default=0)  # synthetic order count

    merchant: Mapped["Merchant"] = relationship(back_populates="menu_items")


class Customer(Base):
    __tablename__ = "customers"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    city_id: Mapped[int] = mapped_column(ForeignKey("cities.id"), index=True)
    dietary_prefs: Mapped[list[str]] = mapped_column(JSON, default=list)
    signup_date: Mapped[datetime] = mapped_column(DateTime)
    behavior_persona: Mapped[str | None] = mapped_column(String(60), nullable=True)

    city: Mapped["City"] = relationship()


class Order(Base):
    __tablename__ = "orders"
    id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), index=True)
    merchant_id: Mapped[int] = mapped_column(ForeignKey("merchants.id"), index=True)
    driver_id: Mapped[int | None] = mapped_column(ForeignKey("drivers.id"), index=True, nullable=True)
    city_id: Mapped[int] = mapped_column(ForeignKey("cities.id"), index=True)
    pickup_zone: Mapped[str] = mapped_column(String(80))
    dropoff_zone: Mapped[str] = mapped_column(String(80))
    # Fine-grained spatial location (10x10 grid per city). Nullable for legacy rows.
    pickup_cell_id: Mapped[int | None] = mapped_column(ForeignKey("grid_cells.id"), index=True, nullable=True)
    dropoff_cell_id: Mapped[int | None] = mapped_column(ForeignKey("grid_cells.id"), index=True, nullable=True)
    total: Mapped[float] = mapped_column(Float)              # SGD/MYR/etc, currency abstracted
    driver_earning: Mapped[float] = mapped_column(Float)    # what the driver netted on this order
    status: Mapped[str] = mapped_column(String(20))         # completed | cancelled
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    risk_score: Mapped[float | None] = mapped_column(Float, index=True, nullable=True)
    risk_decision: Mapped[str | None] = mapped_column(String(10), nullable=True)

    driver: Mapped["Driver"] = relationship(back_populates="orders")


class AuthUser(Base):
    """A user account.

    Identity: email + username are both unique and either can be used to log in.
    Linkage: typed FKs (driver_id / customer_id / merchant_id) replace the old
    raw linked_id — DB enforces integrity now. Admin users have all FKs null.
    """
    __tablename__ = "auth_users"

    id: Mapped[int] = mapped_column(primary_key=True)

    # ---- Identity ------------------------------------------------------
    email: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    phone: Mapped[str | None] = mapped_column(String(40), nullable=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str] = mapped_column(String(160))
    avatar_url: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # ---- Role + typed linkage to persona row ---------------------------
    role: Mapped[str] = mapped_column(String(20), index=True)  # driver | customer | merchant | admin
    driver_id:   Mapped[int | None] = mapped_column(ForeignKey("drivers.id"),    nullable=True, index=True)
    customer_id: Mapped[int | None] = mapped_column(ForeignKey("customers.id"),  nullable=True, index=True)
    merchant_id: Mapped[int | None] = mapped_column(ForeignKey("merchants.id"),  nullable=True, index=True)

    # ---- Account state -------------------------------------------------
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)

    # ---- Audit ---------------------------------------------------------
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_login_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    failed_login_count: Mapped[int] = mapped_column(Integer, default=0)

    # ---- Convenience helpers (not columns) ----------------------------
    @property
    def linked_id(self) -> int | None:
        """Return the persona row id this user is linked to, regardless of role.

        Admin users return None. Kept for backward-compat with older callers.
        """
        return self.driver_id or self.customer_id or self.merchant_id


class UserSession(Base):
    """Active login sessions, one row per cookie.

    We store SHA-256(token) — never the raw signed cookie — so a DB leak
    can't be replayed as a session. Lookup on each request validates the
    cookie's signature first, then verifies the session is still alive here.
    """
    __tablename__ = "user_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    auth_user_id: Mapped[int] = mapped_column(ForeignKey("auth_users.id"), index=True)
    token_hash: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    user_agent: Mapped[str | None] = mapped_column(String(255), nullable=True)
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_used_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class LLMCallLog(Base):
    """Per-invocation log of every LLM call.

    Powers the admin "LLM Performance & Cost" widget and the CloudWatch
    metric pipeline. `hallucinated` flips to True whenever the agent loop
    catches an unknown tool, the supervisor returns an invalid next_agent,
    structured output fails to validate, or the model returns empty content.

    `reasons_json` carries the structured reason codes for forensics, e.g.
    ["unknown_tool:get_city_benchmark", "schema_validation"].
    """
    __tablename__ = "llm_call_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[str] = mapped_column(String(160), index=True)
    provider: Mapped[str] = mapped_column(String(32), index=True)          # "bedrock" | "anthropic-direct"
    agent: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)  # supervisor / driver_success / ...
    auth_user_id: Mapped[int | None] = mapped_column(ForeignKey("auth_users.id"), nullable=True, index=True)
    invoked_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    hallucinated: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    reasons_json: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(String(255), nullable=True)


class LoginAttempt(Base):
    """Audit log of login attempts (success + failure) for forensics + soft rate-limit."""
    __tablename__ = "login_attempts"

    id: Mapped[int] = mapped_column(primary_key=True)
    identifier: Mapped[str] = mapped_column(String(160), index=True)  # what the user typed (username or email)
    success: Mapped[bool] = mapped_column(Boolean)
    error_reason: Mapped[str | None] = mapped_column(String(40), nullable=True)
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class ChatMessage(Base):
    """Persisted chat history per auth user — agents may consult past sessions."""
    __tablename__ = "chat_messages"
    id: Mapped[int] = mapped_column(primary_key=True)
    auth_user_id: Mapped[int] = mapped_column(ForeignKey("auth_users.id"), index=True)
    role: Mapped[str] = mapped_column(String(20))  # user | assistant | system
    agent: Mapped[str | None] = mapped_column(String(40), nullable=True)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class Incentive(Base):
    """Active driver-earning campaigns. The Driver Success Agent reads these."""
    __tablename__ = "incentives"
    id: Mapped[int] = mapped_column(primary_key=True)
    city_id: Mapped[int] = mapped_column(ForeignKey("cities.id"), index=True)
    title: Mapped[str] = mapped_column(String(160))
    description: Mapped[str] = mapped_column(Text)
    vehicle_type: Mapped[str] = mapped_column(String(20))   # bike | car | any
    zone: Mapped[str | None] = mapped_column(String(80), nullable=True)  # null = city-wide
    bonus_amount: Mapped[float] = mapped_column(Float)
    starts_at: Mapped[datetime] = mapped_column(DateTime)
    ends_at: Mapped[datetime] = mapped_column(DateTime)


# =============================================================================
# SPATIAL GRID TABLES
# 10×10 grid per city (100 cells × 5 cities = 500 cells). Backs the cell-level
# DP optimizer in backend/optim/daily_planner.py. Designed to be performant —
# every optimizer call hits cell_demand_hourly (small indexed table), never
# scans raw orders.
# =============================================================================
class GridCell(Base):
    """One cell of a city's 10×10 spatial grid.

    Cells are mapped 1:1 onto one of the city's legacy 4 zones (so we can
    compress consecutive same-zone cells into readable blocks for the UI),
    and tagged with a coarse cell_type that biases demand realism in the seed.
    """
    __tablename__ = "grid_cells"

    id: Mapped[int] = mapped_column(primary_key=True)
    city_id: Mapped[int] = mapped_column(ForeignKey("cities.id"), index=True)
    grid_x: Mapped[int] = mapped_column(Integer)                 # 0-9
    grid_y: Mapped[int] = mapped_column(Integer)                 # 0-9
    center_lat: Mapped[float] = mapped_column(Float)
    center_lon: Mapped[float] = mapped_column(Float)
    zone_name: Mapped[str] = mapped_column(String(80))           # legacy zone (Orchard / Bugis / etc.)
    cell_type: Mapped[str] = mapped_column(String(20))           # commercial | residential | transit | industrial


class CellDemandHourly(Base):
    """Pre-aggregated demand stats per (cell, hour-of-day, day-of-week).

    Recomputed from orders at the end of each seed pass. Optimizer reads this
    as one batched SELECT scoped to the driver's city + plan day-of-week —
    typically <600 rows, returned in single-digit ms.
    """
    __tablename__ = "cell_demand_hourly"

    id: Mapped[int] = mapped_column(primary_key=True)
    cell_id: Mapped[int] = mapped_column(ForeignKey("grid_cells.id"), index=True)
    hour: Mapped[int] = mapped_column(Integer)                   # 0-23
    day_of_week: Mapped[int] = mapped_column(Integer)            # 0=Mon
    orders_total: Mapped[int] = mapped_column(Integer)
    drivers_unique: Mapped[int] = mapped_column(Integer)
    earnings_total: Mapped[float] = mapped_column(Float)
    avg_earning_per_trip: Mapped[float] = mapped_column(Float)


class CellTravelMinutes(Base):
    """Off-peak base travel time between two cells in the same city.

    Computed from Haversine distance × city average speed. Optimizer applies
    a per-hour traffic multiplier at runtime so we don't need to store all
    24 hours per pair. Stored sparsely — only within-city pairs.
    """
    __tablename__ = "cell_travel_minutes"

    id: Mapped[int] = mapped_column(primary_key=True)
    from_cell_id: Mapped[int] = mapped_column(ForeignKey("grid_cells.id"), index=True)
    to_cell_id: Mapped[int] = mapped_column(ForeignKey("grid_cells.id"), index=True)
    base_minutes: Mapped[float] = mapped_column(Float)
    distance_km: Mapped[float] = mapped_column(Float)


# =============================================================================
# DRIVER PLANNER TABLES
# Feeds the DP-based daily route optimizer in backend/optim/daily_planner.py.
# Idle minutes are the optimizer's INTERNAL objective — never surfaced to drivers.
# =============================================================================
class DriverPreferences(Base):
    """One row per driver, captured via /api/driver/preferences.

    The planner reads these as hard/soft constraints:
      - preferred_zones: SOFT — the optimizer biases toward these zones.
      - blackout_hours: HARD — the optimizer never schedules during these hours.
      - weekly_target_sgd: SOFT — used to nudge plan towards meeting weekly goal.
    """
    __tablename__ = "driver_preferences"

    driver_id: Mapped[int] = mapped_column(ForeignKey("drivers.id"), primary_key=True)
    preferred_zones: Mapped[list[str]] = mapped_column(JSON, default=list)
    blackout_hours: Mapped[list[int]] = mapped_column(JSON, default=list)     # 0-23 ints
    weekly_target_sgd: Mapped[float | None] = mapped_column(Float, nullable=True)
    notify_plan_changes: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DriverSchedule(Base):
    """One row per (driver, day-of-week) the driver is active.

    A driver who works Mon-Fri 6 PM-10 PM and Sat 11 AM-11 PM has 6 rows here.
    No row for a day = unavailable that day. The planner uses these to define
    each day's active window and shift bounds.
    """
    __tablename__ = "driver_schedule"

    id: Mapped[int] = mapped_column(primary_key=True)
    driver_id: Mapped[int] = mapped_column(ForeignKey("drivers.id"), index=True)
    day_of_week: Mapped[int] = mapped_column(Integer)         # 0=Mon .. 6=Sun
    start_hour: Mapped[int] = mapped_column(Integer)          # 0-23
    end_hour: Mapped[int] = mapped_column(Integer)            # 0-23 (exclusive: ends BEFORE this hour)


class DriverActiveSession(Base):
    """Open = driver is currently ONLINE taking trips. Closed = OFFLINE.

    The "active toggle" in the driver UI opens/closes a row here. A driver
    may close their day early (end_reason='manual_off') or schedule an
    auto-resume at a later time (resume_at not null) — the latter handles
    'log me off until tomorrow 6 AM'.
    """
    __tablename__ = "driver_active_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    driver_id: Mapped[int] = mapped_column(ForeignKey("drivers.id"), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    end_reason: Mapped[str | None] = mapped_column(String(40), nullable=True)
    # 'manual_off' | 'auto_off_session_end' | 'auto_off_until_resume'
    resume_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class DailyPlan(Base):
    """Persisted optimizer output. One per (driver_id, plan_date) — newest wins.

    plan_json is the structured plan the agent surfaces to the driver:
      [
        {"start_hour": 18, "end_hour": 20, "zone": "Orchard",
         "expected_earnings": 52.0, "rationale": "demand 2.1× supply"},
        {"start_hour": 20, "end_hour": 22, "zone": "Bukit Timah", ...}
      ]
    expected_total_idle_min is INTERNAL — used to track planner quality, never
    shown to drivers in the UI or the agent's reply.
    """
    __tablename__ = "daily_plans"

    id: Mapped[int] = mapped_column(primary_key=True)
    driver_id: Mapped[int] = mapped_column(ForeignKey("drivers.id"), index=True)
    plan_date: Mapped[datetime] = mapped_column(DateTime, index=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    plan_json: Mapped[list] = mapped_column(JSON, default=list)
    expected_total_earnings: Mapped[float] = mapped_column(Float)
    expected_total_idle_min: Mapped[int] = mapped_column(Integer)         # INTERNAL
    naive_baseline_earnings: Mapped[float] = mapped_column(Float)         # "stay in home zone" baseline
    actual_total_earnings: Mapped[float | None] = mapped_column(Float, nullable=True)
