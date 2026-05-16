"""Driver Planner endpoints — status, schedule, preferences, daily plan.

Surfaces the new DriverPreferences / DriverSchedule / DriverActiveSession /
DailyPlan tables to the driver UI. Optimizer integration (DP solver) is a
follow-up phase — this endpoint currently returns a structural stub for the
plan so the UI can render before the optimizer ships.

All endpoints scoped to the logged-in driver via require_driver.
"""
from __future__ import annotations
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status as http_status
from pydantic import BaseModel, Field
from sqlalchemy import select

from backend.db.database import get_session
from backend.db import models as M
from backend.api.auth import get_current_user, CurrentUser
from backend.optim.daily_planner import compute_daily_plan


router = APIRouter(prefix="/api/driver/planner", tags=["driver-planner"])

DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ---------------------------------------------------------------------------
# Auth dep (re-declared so we don't import driver_dash and create cycles)
# ---------------------------------------------------------------------------
def require_driver(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if user.role != "driver":
        raise HTTPException(status_code=http_status.HTTP_403_FORBIDDEN, detail="driver access required")
    return user


# ---------------------------------------------------------------------------
# Pydantic models for request bodies
# ---------------------------------------------------------------------------
class ToggleBody(BaseModel):
    go_active: bool                              # True → open a session; False → close current
    resume_at: datetime | None = None            # optional auto-resume timestamp (only when going offline)
    reason: str | None = Field(None, max_length=40)


class PreferencesBody(BaseModel):
    preferred_zones: list[str] | None = None
    blackout_hours: list[int] | None = None
    weekly_target_sgd: float | None = None
    notify_plan_changes: bool | None = None


class ScheduleRow(BaseModel):
    day_of_week: int = Field(..., ge=0, le=6)
    start_hour: int = Field(..., ge=0, le=23)
    end_hour: int = Field(..., ge=1, le=24)


class ScheduleBody(BaseModel):
    schedule: list[ScheduleRow]                  # full replacement — any day not in this list = off


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_active_session(s, driver_id: int) -> M.DriverActiveSession | None:
    """Open session = ended_at IS NULL. At most one per driver."""
    return s.execute(
        select(M.DriverActiveSession)
        .where(M.DriverActiveSession.driver_id == driver_id, M.DriverActiveSession.ended_at.is_(None))
        .order_by(M.DriverActiveSession.started_at.desc())
        .limit(1)
    ).scalar_one_or_none()


def _serialize_session(sess: M.DriverActiveSession | None) -> dict[str, Any]:
    if not sess:
        return {"is_active": False, "session_id": None}
    return {
        "is_active": True,
        "session_id": sess.id,
        "started_at": sess.started_at.isoformat(),
        "resume_at": sess.resume_at.isoformat() if sess.resume_at else None,
    }


def _today_schedule(s, driver_id: int) -> dict[str, Any] | None:
    """Today's scheduled active window, if any."""
    dow = datetime.utcnow().weekday()
    row = s.execute(
        select(M.DriverSchedule)
        .where(M.DriverSchedule.driver_id == driver_id, M.DriverSchedule.day_of_week == dow)
        .order_by(M.DriverSchedule.start_hour.asc())
        .limit(1)
    ).scalar_one_or_none()
    if not row:
        return None
    return {
        "day_of_week": dow,
        "day_label": DAY_LABELS[dow],
        "start_hour": row.start_hour,
        "end_hour": row.end_hour,
    }


def _ensure_preferences(s, driver_id: int) -> M.DriverPreferences:
    """Get-or-create — handles users registered before the planner tables existed."""
    p = s.get(M.DriverPreferences, driver_id)
    if p is None:
        p = M.DriverPreferences(driver_id=driver_id, preferred_zones=[], blackout_hours=[], weekly_target_sgd=500.0)
        s.add(p)
        s.flush()
    return p


# ---------------------------------------------------------------------------
# GET /api/driver/planner/status
# ---------------------------------------------------------------------------
@router.get("/status")
def status_endpoint(current: CurrentUser = Depends(require_driver)) -> dict[str, Any]:
    """Top-of-dashboard payload: current online state, today's window, home zone."""
    with get_session() as s:
        d = s.get(M.Driver, current.id)
        if not d:
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="driver not found")
        sess = _get_active_session(s, current.id)
        today = _today_schedule(s, current.id)
        return {
            "driver_id": d.id,
            "name": d.name,
            "city": d.city.name,
            "home_zone": d.home_zone,
            "vehicle_type": d.vehicle_type,
            "session": _serialize_session(sess),
            "today_schedule": today,
        }


# ---------------------------------------------------------------------------
# POST /api/driver/planner/toggle
# ---------------------------------------------------------------------------
@router.post("/toggle")
def toggle_active(body: ToggleBody, current: CurrentUser = Depends(require_driver)) -> dict[str, Any]:
    """Flip the online switch. Going active opens a session; going offline closes it."""
    with get_session() as s:
        sess = _get_active_session(s, current.id)

        if body.go_active:
            if sess:
                return {"ok": True, "already_active": True, "session": _serialize_session(sess)}
            new_sess = M.DriverActiveSession(
                driver_id=current.id,
                started_at=datetime.utcnow(),
                ended_at=None,
            )
            s.add(new_sess); s.flush()
            return {"ok": True, "session": _serialize_session(new_sess)}

        # Going offline
        if not sess:
            return {"ok": True, "already_offline": True, "session": _serialize_session(None)}
        sess.ended_at = datetime.utcnow()
        sess.end_reason = body.reason or "manual_off"
        sess.resume_at = body.resume_at
        s.flush()
        return {"ok": True, "session": _serialize_session(None), "ended_session_id": sess.id}


# ---------------------------------------------------------------------------
# GET / PUT /api/driver/planner/preferences
# ---------------------------------------------------------------------------
@router.get("/preferences")
def get_preferences(current: CurrentUser = Depends(require_driver)) -> dict[str, Any]:
    with get_session() as s:
        d = s.get(M.Driver, current.id)
        p = _ensure_preferences(s, current.id)
        schedule = s.execute(
            select(M.DriverSchedule)
            .where(M.DriverSchedule.driver_id == current.id)
            .order_by(M.DriverSchedule.day_of_week, M.DriverSchedule.start_hour)
        ).scalars().all()

        # All zones in the driver's city (for UI multi-select)
        all_zones = list(d.city.zones) if d and d.city else []

        return {
            "preferred_zones": p.preferred_zones or [],
            "blackout_hours": p.blackout_hours or [],
            "weekly_target_sgd": p.weekly_target_sgd,
            "notify_plan_changes": p.notify_plan_changes,
            "all_zones": all_zones,
            "home_zone": d.home_zone if d else None,
            "schedule": [
                {
                    "day_of_week": row.day_of_week,
                    "day_label": DAY_LABELS[row.day_of_week],
                    "start_hour": row.start_hour,
                    "end_hour": row.end_hour,
                }
                for row in schedule
            ],
        }


@router.put("/preferences")
def update_preferences(body: PreferencesBody, current: CurrentUser = Depends(require_driver)) -> dict[str, Any]:
    with get_session() as s:
        p = _ensure_preferences(s, current.id)
        if body.preferred_zones is not None:
            p.preferred_zones = body.preferred_zones
        if body.blackout_hours is not None:
            # de-dupe + clamp
            p.blackout_hours = sorted(set(h for h in body.blackout_hours if 0 <= h <= 23))
        if body.weekly_target_sgd is not None:
            p.weekly_target_sgd = max(0.0, body.weekly_target_sgd)
        if body.notify_plan_changes is not None:
            p.notify_plan_changes = body.notify_plan_changes
        p.updated_at = datetime.utcnow()
        s.flush()
        return {"ok": True}


# ---------------------------------------------------------------------------
# PUT /api/driver/planner/schedule  (full replacement)
# ---------------------------------------------------------------------------
@router.put("/schedule")
def replace_schedule(body: ScheduleBody, current: CurrentUser = Depends(require_driver)) -> dict[str, Any]:
    with get_session() as s:
        # Wipe existing
        existing = s.execute(
            select(M.DriverSchedule).where(M.DriverSchedule.driver_id == current.id)
        ).scalars().all()
        for row in existing:
            s.delete(row)
        s.flush()

        for r in body.schedule:
            if r.end_hour <= r.start_hour:
                continue  # skip invalid
            s.add(M.DriverSchedule(
                driver_id=current.id,
                day_of_week=r.day_of_week,
                start_hour=r.start_hour,
                end_hour=r.end_hour,
            ))
        s.flush()
        return {"ok": True, "saved": len(body.schedule)}


# ---------------------------------------------------------------------------
# GET /api/driver/planner/today — DP-OPTIMIZED DAILY PLAN
# Backed by backend/optim/daily_planner.compute_daily_plan().
# Persists the latest plan per (driver, date) to the daily_plans table.
# ---------------------------------------------------------------------------
@router.get("/today")
def todays_plan(current: CurrentUser = Depends(require_driver)) -> dict[str, Any]:
    return compute_daily_plan(current.id)
