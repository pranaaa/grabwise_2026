"""Tools the Driver Success Agent calls.

Each tool is a small, deterministic SQL query over the seeded data. Keep return
shapes JSON-friendly so the LLM can reason over them and so the activity panel
can display them.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select, func, and_
from langchain_core.tools import tool

from backend.db.database import get_session
from backend.db import models as M
from backend.optim.daily_planner import compute_daily_plan as _compute_daily_plan


# ----------------------------- 1. Driver profile ------------------------------
@tool
def get_driver_profile(driver_id: int) -> dict[str, Any]:
    """Look up a driver's profile (name, city, vehicle, rating, tenure).

    Args:
        driver_id: The driver's numeric ID.
    """
    with get_session() as s:
        d: M.Driver | None = s.get(M.Driver, driver_id)
        if not d:
            return {"error": f"driver {driver_id} not found"}
        return {
            "driver_id": d.id,
            "name": d.name,
            "city": d.city.name,
            "vehicle_type": d.vehicle_type,
            "rating": d.rating,
            "cancel_rate": d.cancel_rate,
            "joined_date": d.joined_date.date().isoformat(),
            "tenure_days": (datetime.utcnow() - d.joined_date).days,
        }


# --------------------------- 2. Earnings rollup -------------------------------
@tool
def get_driver_earnings(driver_id: int, days: int = 7) -> dict[str, Any]:
    """Get a driver's earnings rollup over the last N days.

    Returns daily totals plus an overall summary (total earned, trips, avg/day).

    Args:
        driver_id: The driver's numeric ID.
        days: Window in days (default 7, max 90).
    """
    days = max(1, min(days, 90))
    cutoff = datetime.utcnow() - timedelta(days=days)
    with get_session() as s:
        rows = s.execute(
            select(
                func.date(M.Order.created_at).label("day"),
                func.count(M.Order.id).label("trips"),
                func.coalesce(func.sum(M.Order.driver_earning), 0.0).label("earnings"),
            )
            .where(
                M.Order.driver_id == driver_id,
                M.Order.created_at >= cutoff,
                M.Order.status == "completed",
            )
            .group_by(func.date(M.Order.created_at))
            .order_by(func.date(M.Order.created_at))
        ).all()

        daily = [
            {"date": str(r.day), "trips": int(r.trips), "earnings": round(float(r.earnings), 2)}
            for r in rows
        ]
        total_earnings = round(sum(d["earnings"] for d in daily), 2)
        total_trips = sum(d["trips"] for d in daily)
        return {
            "driver_id": driver_id,
            "window_days": days,
            "total_earnings": total_earnings,
            "total_trips": total_trips,
            "avg_per_day": round(total_earnings / days, 2),
            "daily": daily,
        }


# --------------------------- 3. Busy zones ------------------------------------
@tool
def get_busy_zones(city_name: str, day_of_week: str | None = None, hour: int | None = None) -> dict[str, Any]:
    """Find the top zones by completed-order volume in a city.

    Filters by day-of-week and/or hour-of-day if provided. Use this to recommend
    where a driver should head.

    Args:
        city_name: City name (e.g. "Singapore", "Jakarta").
        day_of_week: Optional, one of "Mon"|"Tue"|...|"Sun".
        hour: Optional, hour-of-day 0-23.
    """
    dow_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
    with get_session() as s:
        city = s.scalar(select(M.City).where(M.City.name == city_name))
        if not city:
            return {"error": f"city {city_name!r} not found"}

        # Recent window (last 30 days) to keep "busy" current.
        cutoff = datetime.utcnow() - timedelta(days=30)
        q = (
            select(M.Order.pickup_zone, func.count(M.Order.id).label("trips"))
            .where(
                M.Order.city_id == city.id,
                M.Order.status == "completed",
                M.Order.created_at >= cutoff,
            )
            .group_by(M.Order.pickup_zone)
            .order_by(func.count(M.Order.id).desc())
        )
        # Filter to day-of-week / hour if specified — done in Python since SQLite's
        # date functions are limited; the dataset is small enough that this is fine.
        rows = s.execute(q).all()
        zones = [{"zone": r.pickup_zone, "trips": int(r.trips)} for r in rows]

        if day_of_week or hour is not None:
            target_dow = dow_map.get(day_of_week) if day_of_week else None
            # Recompute with row-level filter
            recent = s.execute(
                select(M.Order.pickup_zone, M.Order.created_at)
                .where(
                    M.Order.city_id == city.id,
                    M.Order.status == "completed",
                    M.Order.created_at >= cutoff,
                )
            ).all()
            counts: dict[str, int] = {}
            for zone, ts in recent:
                if target_dow is not None and ts.weekday() != target_dow:
                    continue
                if hour is not None and ts.hour != hour:
                    continue
                counts[zone] = counts.get(zone, 0) + 1
            zones = [{"zone": z, "trips": n} for z, n in sorted(counts.items(), key=lambda x: -x[1])]

        return {
            "city": city.name,
            "filters": {"day_of_week": day_of_week, "hour": hour},
            "top_zones": zones[:5],
        }


# --------------------------- 4. Active incentives ----------------------------
@tool
def get_active_incentives(city_name: str, vehicle_type: str | None = None) -> dict[str, Any]:
    """List currently-active driver incentives in a city.

    Args:
        city_name: City name.
        vehicle_type: Optional, "bike" or "car". If omitted, returns all.
    """
    now = datetime.utcnow()
    with get_session() as s:
        city = s.scalar(select(M.City).where(M.City.name == city_name))
        if not city:
            return {"error": f"city {city_name!r} not found"}

        filters = [M.Incentive.city_id == city.id, M.Incentive.starts_at <= now, M.Incentive.ends_at >= now]
        if vehicle_type:
            filters.append(M.Incentive.vehicle_type.in_([vehicle_type, "any"]))

        rows = s.execute(select(M.Incentive).where(and_(*filters))).scalars().all()
        return {
            "city": city.name,
            "count": len(rows),
            "incentives": [
                {
                    "title": r.title,
                    "description": r.description,
                    "vehicle_type": r.vehicle_type,
                    "zone": r.zone,
                    "bonus_amount": r.bonus_amount,
                    "ends_at": r.ends_at.isoformat(),
                }
                for r in rows
            ],
        }


# ============================================================================
#  Deck-aligned tools — one per pillar of the Driver Success Agent
#    Pillar 1: "Peak Earning Window"  → get_peak_earning_windows
#    Pillar 2: "Geo Hotspots"         → predict_demand_hotspots
#    Pillar 3: "Earning Optimization" → get_savings_recommendations
# ============================================================================

_DOW = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}


# --------------------------- 5. Peak earning windows --------------------------
@tool
def get_peak_earning_windows(driver_id: int, day_of_week: str | None = None) -> dict[str, Any]:
    """Find this driver's best time-of-day earning windows over the last 30 days.

    Returns the top 3 three-hour windows ranked by average earnings per session,
    plus how many days/trips back the recommendation. Use this to answer
    "when should I drive?" — the deck's "Peak Earning Window" pillar.

    Args:
        driver_id: Driver's numeric ID.
        day_of_week: Optional, "Mon"|"Tue"|...|"Sun". If set, restricts the
            analysis to that weekday only (e.g. for Friday-night planning).
    """
    target_dow = _DOW.get(day_of_week) if day_of_week else None
    cutoff = datetime.utcnow() - timedelta(days=30)

    with get_session() as s:
        rows = s.execute(
            select(M.Order.created_at, M.Order.driver_earning).where(
                M.Order.driver_id == driver_id,
                M.Order.created_at >= cutoff,
                M.Order.status == "completed",
            )
        ).all()

    rows = [(ts, float(e)) for ts, e in rows if target_dow is None or ts.weekday() == target_dow]
    if not rows:
        return {
            "driver_id": driver_id,
            "day_of_week": day_of_week,
            "top_windows": [],
            "note": "no completed orders in the last 30 days for this filter",
        }

    # Aggregate per hour-of-day: total earnings + the set of dates active at that hour.
    by_hour: dict[int, dict[str, Any]] = {h: {"earn": 0.0, "days": set(), "trips": 0} for h in range(24)}
    for ts, e in rows:
        h = ts.hour
        by_hour[h]["earn"] += e
        by_hour[h]["days"].add(ts.date())
        by_hour[h]["trips"] += 1

    # Slide a 3-hour window across the day; rank windows by avg earnings/session.
    windows: list[dict[str, Any]] = []
    for start in range(0, 22):
        hours = (start, start + 1, start + 2)
        total_earn = sum(by_hour[h]["earn"] for h in hours)
        total_trips = sum(by_hour[h]["trips"] for h in hours)
        active_days: set = set().union(*[by_hour[h]["days"] for h in hours])
        if not active_days:
            continue
        windows.append({
            "window": f"{start:02d}:00-{start + 3:02d}:00",
            "avg_earnings_per_session": round(total_earn / len(active_days), 2),
            "trips_observed": total_trips,
            "days_observed": len(active_days),
        })

    windows.sort(key=lambda w: -w["avg_earnings_per_session"])
    return {
        "driver_id": driver_id,
        "day_of_week": day_of_week,
        "lookback_days": 30,
        "top_windows": windows[:3],
    }


# --------------------------- 6. Demand-hotspot prediction ---------------------
@tool
def predict_demand_hotspots(
    city_name: str,
    day_of_week: str | None = None,
    hour: int | None = None,
) -> dict[str, Any]:
    """Predict which zones in a city will be busiest for a given day-of-week and hour.

    Uses an 8-week historical average over completed orders. The deck calls this
    pillar "Geo Hotspots — predictions of areas with high delivery density."
    If `hour` is given, the window is hour ± 1 for smoothing.

    Args:
        city_name: City name (e.g. "Singapore").
        day_of_week: Optional "Mon"|...|"Sun".
        hour: Optional 0-23.
    """
    target_dow = _DOW.get(day_of_week) if day_of_week else None
    cutoff = datetime.utcnow() - timedelta(days=56)

    with get_session() as s:
        city = s.scalar(select(M.City).where(M.City.name == city_name))
        if not city:
            return {"error": f"city {city_name!r} not found"}

        rows = s.execute(
            select(M.Order.pickup_zone, M.Order.created_at).where(
                M.Order.city_id == city.id,
                M.Order.created_at >= cutoff,
                M.Order.status == "completed",
            )
        ).all()

    def _matches(ts: datetime) -> bool:
        if target_dow is not None and ts.weekday() != target_dow:
            return False
        if hour is not None and not (hour - 1 <= ts.hour <= hour + 1):
            return False
        return True

    matching = [(z, ts) for z, ts in rows if _matches(ts)]
    if not matching:
        return {
            "city": city_name,
            "filters": {"day_of_week": day_of_week, "hour": hour},
            "predictions": [],
            "note": "not enough historical data for this filter",
        }

    counts: dict[str, int] = {}
    dates: set = set()
    for zone, ts in matching:
        counts[zone] = counts.get(zone, 0) + 1
        dates.add(ts.date())

    days_observed = max(1, len(dates))
    predictions = sorted(
        [
            {
                "zone": z,
                "expected_orders_per_relevant_day": round(c / days_observed, 2),
                "historical_orders": c,
            }
            for z, c in counts.items()
        ],
        key=lambda x: -x["historical_orders"],
    )

    return {
        "city": city_name,
        "filters": {"day_of_week": day_of_week, "hour": hour},
        "lookback_days": 56,
        "days_observed": days_observed,
        "predictions": predictions[:5],
    }


# --------------------------- 7. Savings recommendations ----------------------
@tool
def get_savings_recommendations(driver_id: int, weeks: int = 4) -> dict[str, Any]:
    """Compute earnings-optimization signals for the 'Earning Optimization' pillar.

    Returns: total/avg earnings, idle hours-of-day, cancel-rate cost, and
    best/worst hour gap. The agent narrates these as savings-style suggestions
    (e.g. "your 3am-5am hours are unpaid — cutting them saves 2h/shift").

    Args:
        driver_id: Driver's numeric ID.
        weeks: Lookback window in weeks (default 4, max 12).
    """
    weeks = max(1, min(weeks, 12))
    days = weeks * 7
    cutoff = datetime.utcnow() - timedelta(days=days)

    with get_session() as s:
        d: M.Driver | None = s.get(M.Driver, driver_id)
        if not d:
            return {"error": f"driver {driver_id} not found"}

        all_orders = s.execute(
            select(
                M.Order.created_at, M.Order.driver_earning, M.Order.status, M.Order.total
            ).where(M.Order.driver_id == driver_id, M.Order.created_at >= cutoff)
        ).all()
        cancel_rate_stored = d.cancel_rate

    completed = [(ts, float(e)) for ts, e, st, _ in all_orders if st == "completed"]
    cancellations = [t for _, _, st, t in all_orders if st == "cancelled"]

    if not completed:
        return {"error": f"driver {driver_id} has no completed orders in the last {days} days"}

    total_earn = sum(e for _, e in completed)
    total_trips = len(completed)
    avg_per_trip = total_earn / total_trips

    # Per-day rollup
    by_day: dict[Any, float] = {}
    for ts, e in completed:
        by_day[ts.date()] = by_day.get(ts.date(), 0.0) + e
    days_active = len(by_day)
    avg_per_active_day = total_earn / max(1, days_active)

    # Per-hour earnings
    by_hour = {h: 0.0 for h in range(24)}
    for ts, e in completed:
        by_hour[ts.hour] += e
    nonzero = [(h, v) for h, v in by_hour.items() if v > 0]
    best_hour, best_earn = max(nonzero, key=lambda x: x[1])
    worst_hour, worst_earn = min(nonzero, key=lambda x: x[1])
    idle_hours = [h for h, v in by_hour.items() if v == 0]

    # Estimated earnings lost to cancellations (cancelled trips × avg earning)
    cancel_count = len(cancellations)
    estimated_lost = round(cancel_count * avg_per_trip, 2)

    return {
        "driver_id": driver_id,
        "lookback_days": days,
        "total_earnings": round(total_earn, 2),
        "completed_trips": total_trips,
        "active_days": days_active,
        "avg_per_active_day": round(avg_per_active_day, 2),
        "avg_per_trip": round(avg_per_trip, 2),
        "best_hour": {"hour": best_hour, "total_earnings": round(best_earn, 2)},
        "worst_hour_with_activity": {"hour": worst_hour, "total_earnings": round(worst_earn, 2)},
        "idle_hours_of_day": sorted(idle_hours),
        "cancellations": cancel_count,
        "estimated_earnings_lost_to_cancellations": estimated_lost,
        "stored_cancel_rate": cancel_rate_stored,
    }


# ============================================================================
#  Cross-agent tool — used at the end of the customer→fraud→driver chain
#  to assign a driver to a freshly-approved order.
# ============================================================================
@tool
def match_driver_for_order(
    city_name: str,
    pickup_zone: str | None = None,
    late_night: bool = False,
    vehicle_type: str | None = None,
) -> dict[str, Any]:
    """Pick the best available driver for an order.

    Selection rules:
      - Active drivers in the requested city only.
      - If late_night=True: rating ≥ 4.7 and cancel_rate ≤ 0.05 (the
        Trusted Late-Night Matching pillar).
      - Otherwise: rating ≥ 4.3 and cancel_rate ≤ 0.10.
      - Sorted by rating desc, then cancel_rate asc.
      - Returns the top match plus 2 alternates.

    Args:
        city_name: City of the merchant/pickup.
        pickup_zone: Optional zone to bias toward (we still return any active
            driver in the city — zone preference is a soft hint).
        late_night: True if the order is being placed late at night.
        vehicle_type: Optional, "bike" or "car".
    """
    from sqlalchemy import select, and_
    with get_session() as s:
        city = s.scalar(select(M.City).where(M.City.name == city_name))
        if not city:
            return {"error": f"city {city_name!r} not found"}

        if late_night:
            min_rating, max_cancel = 4.7, 0.05
            criteria_label = "late-night (Trusted Matching)"
        else:
            min_rating, max_cancel = 4.3, 0.10
            criteria_label = "standard"

        filters = [
            M.Driver.city_id == city.id,
            M.Driver.is_active.is_(True),
            M.Driver.rating >= min_rating,
            M.Driver.cancel_rate <= max_cancel,
        ]
        if vehicle_type:
            filters.append(M.Driver.vehicle_type == vehicle_type)

        candidates = s.execute(
            select(M.Driver)
            .where(and_(*filters))
            .order_by(M.Driver.rating.desc(), M.Driver.cancel_rate.asc())
            .limit(3)
        ).scalars().all()

        if not candidates:
            return {
                "city": city_name,
                "criteria": criteria_label,
                "matched_driver": None,
                "alternates": [],
                "note": "no active drivers met the trust criteria — relax filters or escalate",
            }

        def _info(d: M.Driver) -> dict[str, Any]:
            return {
                "driver_id": d.id,
                "name": d.name,
                "vehicle_type": d.vehicle_type,
                "rating": d.rating,
                "cancel_rate": d.cancel_rate,
            }

        return {
            "city": city_name,
            "pickup_zone": pickup_zone,
            "criteria": criteria_label,
            "matched_driver": _info(candidates[0]),
            "alternates": [_info(d) for d in candidates[1:]],
        }


# ============================================================================
#  Cross-agent tool — used in the merchant→driver chain when a merchant
#  asks "will I have drivers covering my orders?"
# ============================================================================
@tool
def estimate_driver_availability(
    city_name: str,
    zone: str | None = None,
    vehicle_type: str | None = None,
) -> dict[str, Any]:
    """Estimate active driver supply in a city + concentration in a specific zone.

    Used in cross-agent flows when a Merchant asks about driver coverage for
    expected demand. Returns total active drivers in the city, plus (if a zone
    is given) the share of recent pickups originating in that zone — a proxy
    for how many drivers tend to be reachable there.

    Args:
        city_name: City name.
        zone: Optional zone (one of the city's known zones).
        vehicle_type: Optional, "bike" or "car".
    """
    from sqlalchemy import select, func, and_
    cutoff = datetime.utcnow() - timedelta(days=30)
    with get_session() as s:
        city = s.scalar(select(M.City).where(M.City.name == city_name))
        if not city:
            return {"error": f"city {city_name!r} not found"}

        # Total active drivers in city (filtered by vehicle_type)
        d_filters = [M.Driver.city_id == city.id, M.Driver.is_active.is_(True)]
        if vehicle_type:
            d_filters.append(M.Driver.vehicle_type == vehicle_type)
        total_active = s.scalar(select(func.count(M.Driver.id)).where(and_(*d_filters))) or 0

        result: dict[str, Any] = {
            "city": city_name,
            "vehicle_type": vehicle_type,
            "total_active_drivers": total_active,
        }

        if zone:
            zone_orders = s.scalar(
                select(func.count(M.Order.id)).where(
                    M.Order.city_id == city.id,
                    M.Order.pickup_zone == zone,
                    M.Order.created_at >= cutoff,
                    M.Order.status == "completed",
                )
            ) or 0
            city_orders = s.scalar(
                select(func.count(M.Order.id)).where(
                    M.Order.city_id == city.id,
                    M.Order.created_at >= cutoff,
                    M.Order.status == "completed",
                )
            ) or 0
            zone_pct = (zone_orders / city_orders * 100) if city_orders else 0.0
            result.update({
                "zone": zone,
                "zone_concentration_pct": round(zone_pct, 1),
                "zone_orders_30d": zone_orders,
                "estimated_zone_drivers": round(total_active * zone_pct / 100, 1),
            })

        return result


# ---------------------------------------------------------------------------
# 10. Daily plan (DP optimizer) — the primary tool for any "plan my day",
# "where should I drive", "when should I work", or route-optimization question.
# Returns a structured plan with per-block expected earnings + rationale + an
# uplift % vs the naive "stay in one zone all shift" baseline.
# If the driver is off today, automatically plans the next scheduled day.
# ---------------------------------------------------------------------------
@tool
def generate_daily_plan(driver_id: int) -> dict[str, Any]:
    """Generate a DP-optimized daily route plan for the driver.

    The plan balances three things — expected earnings per (zone, hour),
    a soft bonus for the driver's preferred zones, and a travel-time penalty
    between zones — by solving a backward dynamic-programming recursion over
    the (zone, hour) lattice for the driver's scheduled active window.

    Always call this as the **first** tool for any planning, zone-selection,
    or "when should I drive" question. It already aggregates demand + supply
    from history internally — you don't need to call get_peak_earning_windows,
    get_busy_zones, or predict_demand_hotspots separately.

    Args:
        driver_id: The driver's numeric ID.

    Returns:
        {
          "available": bool,
          "plan_date": "YYYY-MM-DD",
          "is_today": bool,                          # False if planned for a future day
          "day_label": "Monday",                     # which weekday the plan is for
          "summary": {
            "expected_total_earnings": 81.42,
            "naive_baseline_earnings": 68.10,
            "uplift_pct": 19.6                       # how much the plan beats "stay in home zone"
          },
          "blocks": [
            {"start_hour": 8, "end_hour": 11, "zone": "Orchard", "expected_earnings": 24.10,
             "rationale": "preferred zone · steady demand"},
            ...
          ]
        }

    If unavailable, returns {"available": False, "message": "..."}.
    """
    return _compute_daily_plan(driver_id)


DRIVER_TOOLS = [
    get_driver_profile,
    get_driver_earnings,
    generate_daily_plan,
    get_busy_zones,
    get_active_incentives,
    get_peak_earning_windows,
    predict_demand_hotspots,
    get_savings_recommendations,
    match_driver_for_order,
    estimate_driver_availability,
]
