"""Driver Daily Plan Optimizer — cell-level backward DP.

State space
-----------
(cell_id, hour_index) for each hour in the driver's scheduled active window.
At 100 cells × ~10 hours = ~1,000 states — small enough that backward DP
gives us the *exact optimum* in ~50-100 ms per call. No A* needed.

Bellman recursion
-----------------
    V*[c, h_max] = 0
    V*[c, h]     = max over next_c of:
                     reward(c, h, next_c) + V*[next_c, h+1]

    reward(c, h, next_c) =
        earning_rate(c, h)                              # per-hour earnings in c at hour h
        × (1 - travel_loss(c, next_c, h))               # fraction of hour preserved after moving
        × (PREFERRED_BONUS if c in preferred_zones)     # soft preference bonus
        - fatigue_penalty(consecutive_active_hours)     # quadratic in consecutive hours

Data sources
------------
- `cell_demand_hourly`: pre-aggregated per (cell, hour, day_of_week). Optimizer
  loads a single batch for the driver's city + plan day (≤2400 rows max → fast).
- `cell_travel_minutes`: pre-computed Haversine base minutes per cell pair in
  the same city. Optimizer applies an hourly traffic multiplier at runtime.
- `grid_cells`: spatial layout + zone_name for block compression in the UI.

Output
------
Compatible response shape with the previous zone-level optimizer:

    {
      "available": bool, "stub": False, "plan_date": "YYYY-MM-DD",
      "is_today": bool, "day_label": "Monday",
      "summary": {expected_total_earnings, naive_baseline_earnings, uplift_pct},
      "blocks": [{start_hour, end_hour, zone, expected_earnings, rationale}, ...]
    }

Consecutive same-zone cells are compressed into one block for readable display
(the optimizer may pick cells (12, 13, 14) all inside Orchard → user sees one
"Orchard 6-9 PM" block, even though the underlying solve was per-cell).
"""
from __future__ import annotations
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import select

from backend.db.database import get_session
from backend.db import models as M

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------------------
# Tunable weights (kept simple and explainable)
# ---------------------------------------------------------------------------
PREFERRED_ZONE_BONUS = 1.10            # 10% earning bonus when in a preferred zone
EARNING_FLOOR_PER_HOUR = 3.0           # floor so empty cells don't break DP
MAX_TRAVEL_LOSS_FRACTION = 0.50        # cap travel loss at half the next hour
FATIGUE_WEIGHT = 0.0                   # disabled by default; non-zero penalizes long shifts


def _traffic_multiplier(hour: int) -> float:
    """Multiplier on base travel minutes by hour-of-day. >1 = slower (more traffic)."""
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        return 1.40           # AM/PM rush
    if hour <= 5 or hour >= 22:
        return 0.80           # late night / early morning
    return 1.05               # daytime baseline


# ---------------------------------------------------------------------------
# Load aggregates for the planning window
# ---------------------------------------------------------------------------
def _load_demand(s, city_id: int, dow: int) -> dict[int, dict[int, dict[str, float]]]:
    """{cell_id: {hour: {orders_total, drivers_unique, avg_earning_per_trip}}}"""
    rows = s.execute(
        select(M.CellDemandHourly, M.GridCell)
        .join(M.GridCell, M.CellDemandHourly.cell_id == M.GridCell.id)
        .where(M.GridCell.city_id == city_id, M.CellDemandHourly.day_of_week == dow)
    ).all()
    out: dict[int, dict[int, dict[str, float]]] = defaultdict(dict)
    for d, _cell in rows:
        out[d.cell_id][d.hour] = {
            "orders_total": float(d.orders_total),
            "drivers_unique": float(max(d.drivers_unique, 1)),
            "avg_earning_per_trip": float(d.avg_earning_per_trip),
        }
    return out


def _load_travel(s, city_id: int) -> dict[tuple[int, int], float]:
    """{(from_cell, to_cell): base_minutes}"""
    rows = s.execute(
        select(M.CellTravelMinutes, M.GridCell)
        .join(M.GridCell, M.CellTravelMinutes.from_cell_id == M.GridCell.id)
        .where(M.GridCell.city_id == city_id)
    ).all()
    return {(r.from_cell_id, r.to_cell_id): float(r.base_minutes) for r, _ in rows}


def _load_cells(s, city_id: int) -> list[M.GridCell]:
    return s.execute(
        select(M.GridCell)
        .where(M.GridCell.city_id == city_id)
        .order_by(M.GridCell.grid_y, M.GridCell.grid_x)
    ).scalars().all()


# ---------------------------------------------------------------------------
# Reward function (per state, per next-state choice)
#
# Temporal smoothing: instead of using only (cell, exact_hour) — which can be
# noisy with sparse data — we average the per-driver-earning rate over a
# 3-hour rolling window {hour-1, hour, hour+1}. This is equivalent to a
# uniform-kernel smoothing along the time dimension and trades a small bias
# at hour boundaries for a *much* lower variance estimate at sparse cells.
# ---------------------------------------------------------------------------
SMOOTH_HOUR_OFFSETS = (-1, 0, 1)


def _earning_rate(
    cell_id: int,
    hour: int,
    demand: dict[int, dict[int, dict[str, float]]],
    zone_name: str,
    preferred_zones: set[str],
) -> tuple[float, float]:
    """Return (rate_per_hour, pressure_ratio) for one driver in (cell, hour).

    Smoothed over a 3-hour window for robustness.
    """
    cell_buckets = demand.get(cell_id, {})
    rates: list[float] = []
    pressures: list[float] = []
    for off in SMOOTH_HOUR_OFFSETS:
        target_h = (hour + off) % 24
        cd = cell_buckets.get(target_h)
        if not cd:
            continue
        orders = cd["orders_total"]
        drivers = max(cd["drivers_unique"], 1)
        avg = cd["avg_earning_per_trip"] or 15.0
        per_driver = orders / drivers
        rates.append(per_driver * avg)
        pressures.append(per_driver)

    if not rates:
        return (EARNING_FLOOR_PER_HOUR, 0.0)

    # Centre hour weighted slightly more so we don't blur peaks too much
    weights = [1.0 if off == 0 else 0.7 for off in SMOOTH_HOUR_OFFSETS][: len(rates)]
    rate = sum(r * w for r, w in zip(rates, weights)) / sum(weights)
    pressure = sum(pressures) / len(pressures)

    if zone_name in preferred_zones:
        rate *= PREFERRED_ZONE_BONUS
    return (max(EARNING_FLOOR_PER_HOUR, rate), pressure)


def _travel_loss(
    from_cell: int,
    to_cell: int,
    hour: int,
    base_minutes: dict[tuple[int, int], float],
) -> float:
    """Fraction of the *next* hour's earnings lost to traveling between cells."""
    if from_cell == to_cell:
        return 0.0
    base = base_minutes.get((from_cell, to_cell))
    if base is None:
        # No edge stored — assume far; lose the cap
        return MAX_TRAVEL_LOSS_FRACTION
    travel = base * _traffic_multiplier(hour)
    return min(MAX_TRAVEL_LOSS_FRACTION, travel / 60.0)


# ---------------------------------------------------------------------------
# Pick the "home cell": one cell within the driver's home_zone, preferring commercial
# ---------------------------------------------------------------------------
def _pick_home_cell(cells: list[M.GridCell], home_zone: str | None) -> M.GridCell | None:
    if not cells:
        return None
    candidates = [c for c in cells if c.zone_name == home_zone] if home_zone else cells
    if not candidates:
        candidates = cells
    # Prefer commercial cells (busier), else any in that zone
    commercial = [c for c in candidates if c.cell_type == "commercial"]
    return (commercial or candidates)[0]


# ---------------------------------------------------------------------------
# Rationale string per block
# ---------------------------------------------------------------------------
def _block_rationale(
    zone_name: str,
    avg_pressure: float,
    preferred_zones: set[str],
    travel_to_next: bool,
    next_zone: str | None,
    travel_mins: float,
) -> str:
    parts: list[str] = []
    if zone_name in preferred_zones:
        parts.append("preferred zone")
    if avg_pressure >= 2.0:
        parts.append(f"high demand ({avg_pressure:.1f}× supply)")
    elif avg_pressure >= 1.0:
        parts.append("steady demand")
    else:
        parts.append("low demand")
    if travel_to_next and next_zone and next_zone != zone_name:
        parts.append(f"hop ~{int(round(travel_mins))} min to {next_zone}")
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# Main entry: compute_daily_plan
# ---------------------------------------------------------------------------
def compute_daily_plan(driver_id: int, plan_date: date | None = None) -> dict[str, Any]:
    request_date = plan_date or datetime.utcnow().date()

    with get_session() as s:
        driver = s.get(M.Driver, driver_id)
        if not driver:
            return {"available": False, "message": f"Driver {driver_id} not found", "stub": False}

        if not driver.city:
            return {"available": False, "message": "Driver has no city assigned.", "stub": False}

        # ---- Find a schedule (today or next 7 days) ----
        sched = None
        planned_date = request_date
        for offset in range(0, 8):
            try_date = request_date + timedelta(days=offset)
            try_dow = try_date.weekday()
            sched = s.execute(
                select(M.DriverSchedule)
                .where(M.DriverSchedule.driver_id == driver_id, M.DriverSchedule.day_of_week == try_dow)
                .order_by(M.DriverSchedule.start_hour.asc())
                .limit(1)
            ).scalar_one_or_none()
            if sched:
                planned_date = try_date
                break

        if not sched:
            return {
                "available": False,
                "stub": False,
                "message": "You have no scheduled hours in the next week. Set up your weekly schedule first.",
                "plan_date": request_date.isoformat(),
            }

        dow = planned_date.weekday()
        is_today = (planned_date == request_date)

        # ---- Preferences ----
        prefs = s.get(M.DriverPreferences, driver_id)
        preferred_zones = set(prefs.preferred_zones or []) if prefs else set()
        blackout = set(prefs.blackout_hours or []) if prefs else set()

        hours = [h for h in range(sched.start_hour, sched.end_hour) if h not in blackout]
        if not hours:
            return {
                "available": False, "stub": False,
                "message": "All your active hours today are in blackout.",
                "plan_date": planned_date.isoformat(),
            }

        # ---- Load cell-level data ----
        cells = _load_cells(s, driver.city_id)
        if len(cells) < 2:
            return {"available": False, "stub": False, "message": "No grid cells configured for this city. Re-seed required.", "plan_date": planned_date.isoformat()}

        cell_zone = {c.id: c.zone_name for c in cells}
        cell_ids = [c.id for c in cells]
        demand = _load_demand(s, driver.city_id, dow)
        travel = _load_travel(s, driver.city_id)

        # ---- Backward DP over (cell, hour_index) ----
        n = len(hours)
        # V[i][cell_id] = best total earnings from hour-index i onwards, starting in cell_id
        V: list[dict[int, float]] = [dict() for _ in range(n + 1)]
        nxt: list[dict[int, int]] = [dict() for _ in range(n + 1)]
        for cid in cell_ids:
            V[n][cid] = 0.0

        for i in range(n - 1, -1, -1):
            h = hours[i]
            for cur_cid in cell_ids:
                cur_zone = cell_zone[cur_cid]
                cur_rate, _ = _earning_rate(cur_cid, h, demand, cur_zone, preferred_zones)
                best_val = -1e18
                best_nz = cur_cid
                next_h = hours[i + 1] if i + 1 < n else None
                for nz_cid in cell_ids:
                    if nz_cid == cur_cid:
                        this_earn = cur_rate
                    else:
                        # Pre-compute travel loss applied to the NEXT hour's earnings
                        if next_h is not None:
                            nz_zone = cell_zone[nz_cid]
                            nz_rate, _ = _earning_rate(nz_cid, next_h, demand, nz_zone, preferred_zones)
                            loss = _travel_loss(cur_cid, nz_cid, h, travel) * nz_rate
                            this_earn = cur_rate - loss
                        else:
                            this_earn = cur_rate  # last hour — no next-hour penalty
                    candidate = this_earn + V[i + 1][nz_cid]
                    if candidate > best_val:
                        best_val = candidate
                        best_nz = nz_cid
                V[i][cur_cid] = best_val
                nxt[i][cur_cid] = best_nz

        # ---- Forward trace from home cell ----
        home_cell = _pick_home_cell(cells, driver.home_zone)
        start_cid = home_cell.id if home_cell else cell_ids[0]
        plan_hours: list[dict[str, Any]] = []
        cur_cid = start_cid
        for i in range(n):
            h = hours[i]
            zone = cell_zone[cur_cid]
            rate, pressure = _earning_rate(cur_cid, h, demand, zone, preferred_zones)
            nz_cid = nxt[i][cur_cid]
            if nz_cid != cur_cid and i + 1 < n:
                nz_zone = cell_zone[nz_cid]
                nz_rate, _ = _earning_rate(nz_cid, hours[i + 1], demand, nz_zone, preferred_zones)
                this_earn = max(0.0, rate - _travel_loss(cur_cid, nz_cid, h, travel) * nz_rate)
                travel_mins = travel.get((cur_cid, nz_cid), 0.0) * _traffic_multiplier(h)
            else:
                this_earn = rate
                travel_mins = 0.0
            plan_hours.append({
                "hour": h,
                "cell_id": cur_cid,
                "zone": zone,
                "earnings": this_earn,
                "pressure": pressure,
                "next_cell_id": nz_cid,
                "travel_mins": travel_mins,
            })
            cur_cid = nz_cid

        # ---- Compress consecutive same-ZONE blocks (for UI readability) ----
        blocks: list[dict[str, Any]] = []
        for ph in plan_hours:
            if blocks and blocks[-1]["zone"] == ph["zone"]:
                blocks[-1]["end_hour"] = ph["hour"] + 1
                blocks[-1]["expected_earnings"] += ph["earnings"]
                blocks[-1]["_pressure_sum"] += ph["pressure"]
                blocks[-1]["_hours"] += 1
                blocks[-1]["_outbound_travel"] = ph["travel_mins"]
                blocks[-1]["_outbound_next_zone"] = cell_zone.get(ph["next_cell_id"], ph["zone"])
            else:
                blocks.append({
                    "start_hour": ph["hour"],
                    "end_hour": ph["hour"] + 1,
                    "zone": ph["zone"],
                    "expected_earnings": ph["earnings"],
                    "_pressure_sum": ph["pressure"],
                    "_hours": 1,
                    "_outbound_travel": ph["travel_mins"],
                    "_outbound_next_zone": cell_zone.get(ph["next_cell_id"], ph["zone"]),
                })

        cleaned: list[dict[str, Any]] = []
        for idx, b in enumerate(blocks):
            avg_pressure = b["_pressure_sum"] / max(b["_hours"], 1)
            is_last = idx == len(blocks) - 1
            next_zone = b["_outbound_next_zone"] if not is_last else None
            travel_mins = b["_outbound_travel"] if (next_zone and next_zone != b["zone"]) else 0.0
            cleaned.append({
                "start_hour": b["start_hour"],
                "end_hour":   b["end_hour"],
                "zone":       b["zone"],
                "expected_earnings": round(max(0.0, b["expected_earnings"]), 2),
                "rationale":  _block_rationale(
                    b["zone"], avg_pressure, preferred_zones,
                    travel_to_next=(travel_mins > 0),
                    next_zone=next_zone,
                    travel_mins=travel_mins,
                ),
            })

        total_earn = sum(b["expected_earnings"] for b in cleaned)

        # ---- Naive baseline: stay in home cell all shift ----
        naive_total = 0.0
        home_zone_name = cell_zone[start_cid]
        for h in hours:
            r, _ = _earning_rate(start_cid, h, demand, home_zone_name, preferred_zones)
            naive_total += r
        uplift_pct = (total_earn - naive_total) / max(naive_total, 1e-3) * 100.0

        # ---- Persist plan (upsert per driver+date) ----
        existing = s.execute(
            select(M.DailyPlan)
            .where(
                M.DailyPlan.driver_id == driver_id,
                M.DailyPlan.plan_date == datetime.combine(planned_date, datetime.min.time()),
            )
            .order_by(M.DailyPlan.generated_at.desc())
            .limit(1)
        ).scalar_one_or_none()

        # Internal idle estimate (never surfaced)
        expected_idle_min = max(0, int((len(hours) * 60) * (1.0 - min(1.0, total_earn / (naive_total * 2 + 1e-6)))))

        if existing is None:
            s.add(M.DailyPlan(
                driver_id=driver_id,
                plan_date=datetime.combine(planned_date, datetime.min.time()),
                generated_at=datetime.utcnow(),
                plan_json=cleaned,
                expected_total_earnings=round(total_earn, 2),
                expected_total_idle_min=expected_idle_min,
                naive_baseline_earnings=round(naive_total, 2),
            ))
        else:
            existing.generated_at = datetime.utcnow()
            existing.plan_json = cleaned
            existing.expected_total_earnings = round(total_earn, 2)
            existing.expected_total_idle_min = expected_idle_min
            existing.naive_baseline_earnings = round(naive_total, 2)

        return {
            "available": True,
            "stub": False,
            "plan_date": planned_date.isoformat(),
            "is_today": is_today,
            "day_label": DAY_NAMES[dow],
            "summary": {
                "expected_total_earnings": round(total_earn, 2),
                "naive_baseline_earnings": round(naive_total, 2),
                "uplift_pct": round(uplift_pct, 1),
            },
            "blocks": cleaned,
        }
