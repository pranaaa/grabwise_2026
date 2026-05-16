"""Driver-only personal dashboard endpoints.

Mirrors `backend/api/admin.py` patterns. All endpoints scoped to the logged-in
driver via `require_driver`. Peer comparisons are restricted to other active
drivers with the same city + vehicle type.
"""
from __future__ import annotations
from datetime import datetime, timedelta, date
from collections import defaultdict
from statistics import median
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status as http_status
from sqlalchemy import select

_range = range  # preserve the builtin so endpoints can use a `range` query param.

from backend.db.database import get_session
from backend.db import models as M
from backend.api.auth import get_current_user, CurrentUser
from backend.tools._risk_math import compute_driver_trust_score


router = APIRouter(prefix="/api/driver", tags=["driver"])


# ---------------------------------------------------------------------------
# Auth dep
# ---------------------------------------------------------------------------

def require_driver(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if user.role != "driver":
        raise HTTPException(status_code=http_status.HTTP_403_FORBIDDEN, detail="driver access required")
    return user


def _get_driver_row(s, user: CurrentUser) -> M.Driver:
    d = s.get(M.Driver, user.id)
    if not d:
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="driver record not found")
    return d


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_currency(v: float) -> str:
    if v >= 1_000_000:
        return f"${v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"${v / 1_000:.1f}K"
    return f"${v:.2f}"


def _fmt_int(v: float) -> str:
    return f"{int(round(v))}"


def _fmt_percent(v: float) -> str:
    return f"{v * 100:.1f}%"


def _tenure_label(days: int) -> str:
    days = max(0, int(days))
    years = days // 365
    months = (days % 365) // 30
    if years <= 0 and months <= 0:
        return f"{days}d"
    if years <= 0:
        return f"{months}m"
    return f"{years}y {months}m"


def _persona_label(persona: str | None) -> str | None:
    if not persona:
        return None
    return " ".join(p.capitalize() for p in persona.replace("_", "-").split("-"))


def _ends_in_label(now: datetime, ends_at: datetime) -> str:
    delta = ends_at - now
    if delta.total_seconds() <= 0:
        return "Ended"
    days = delta.days
    hours = (delta.seconds // 3600)
    if days > 0:
        return f"{days}d {hours}h"
    if hours > 0:
        mins = (delta.seconds % 3600) // 60
        return f"{hours}h {mins}m"
    mins = max(1, delta.seconds // 60)
    return f"{mins}m"


def _pct_delta(curr: float, prev: float) -> float:
    if prev == 0:
        return 0.0 if curr == 0 else 100.0
    return round(((curr - prev) / prev) * 100, 1)


def _signed_delta(curr: float, prev: float) -> float:
    return round(curr - prev, 1)


# ---------------------------------------------------------------------------
# /me
# ---------------------------------------------------------------------------

@router.get("/me")
def me(current: CurrentUser = Depends(require_driver)) -> dict[str, Any]:
    now = datetime.utcnow()
    with get_session() as s:
        d = _get_driver_row(s, current)
        tenure_days = max(0, (now - d.joined_date).days)
        trust_score, components, _reasons = compute_driver_trust_score(
            rating=d.rating,
            cancel_rate=d.cancel_rate,
            tenure_days=tenure_days,
        )
        return {
            "id": d.id,
            "name": d.name,
            "city": d.city.name if d.city else "",
            "city_id": d.city_id,
            "vehicle_type": d.vehicle_type,
            "rating": round(d.rating, 2),
            "cancel_rate": round(d.cancel_rate, 4),
            "joined_date": d.joined_date.isoformat(),
            "tenure_days": tenure_days,
            "tenure_label": _tenure_label(tenure_days),
            "behavior_persona": d.behavior_persona,
            "behavior_persona_label": _persona_label(d.behavior_persona),
            "trust_score": trust_score,
            "trust_components": components,
        }


# ---------------------------------------------------------------------------
# /kpis
# ---------------------------------------------------------------------------

@router.get("/kpis")
def kpis(current: CurrentUser = Depends(require_driver)) -> dict[str, Any]:
    now = datetime.utcnow()
    cutoff_now = now - timedelta(days=7)
    cutoff_prev_start = now - timedelta(days=14)
    cutoff_30 = now - timedelta(days=30)
    cutoff_prev_30 = now - timedelta(days=60)
    cutoff_spark = now - timedelta(days=14)
    # We need 20 days of data so the 14-day rolling-7d cancel sparkline has history.
    cutoff_cancel_spark = now - timedelta(days=20)

    with get_session() as s:
        d = _get_driver_row(s, current)
        tenure_days = max(0, (now - d.joined_date).days)

        # Pull 60d of orders for this driver (covers 30d + prior-30d).
        rows_60 = s.execute(
            select(M.Order).where(
                M.Order.driver_id == d.id,
                M.Order.created_at >= cutoff_prev_30,
            )
        ).scalars().all()

        completed_now = [o for o in rows_60 if o.status == "completed" and o.created_at >= cutoff_now]
        completed_prev = [
            o for o in rows_60
            if o.status == "completed" and cutoff_prev_start <= o.created_at < cutoff_now
        ]

        all_30 = [o for o in rows_60 if o.created_at >= cutoff_30]
        all_prev_30 = [o for o in rows_60 if cutoff_prev_30 <= o.created_at < cutoff_30]

        earnings_now = sum((o.driver_earning or 0.0) for o in completed_now)
        earnings_prev = sum((o.driver_earning or 0.0) for o in completed_prev)
        trips_now = len(completed_now)
        trips_prev = len(completed_prev)
        avg_now = (earnings_now / trips_now) if trips_now else 0.0
        avg_prev = (earnings_prev / trips_prev) if trips_prev else 0.0

        def cancel_rate(orders: list[M.Order]) -> float:
            cancelled = [o for o in orders if o.status == "cancelled"]
            completed = [o for o in orders if o.status == "completed"]
            denom = len(cancelled) + len(completed)
            return (len(cancelled) / denom) if denom else 0.0

        cancel_30 = cancel_rate(all_30)
        cancel_prev_30 = cancel_rate(all_prev_30)

        # Sparklines: last 14 daily points, oldest -> newest.
        days_14 = [(now - timedelta(days=i)).date() for i in range(13, -1, -1)]
        spark_rows = [o for o in rows_60 if o.created_at >= cutoff_spark]
        by_day_completed: dict[date, list[M.Order]] = defaultdict(list)
        for o in spark_rows:
            if o.status == "completed":
                by_day_completed[o.created_at.date()].append(o)
        spark_earnings = [
            round(sum((o.driver_earning or 0.0) for o in by_day_completed.get(d_, [])), 2)
            for d_ in days_14
        ]
        spark_trips = [len(by_day_completed.get(d_, [])) for d_ in days_14]
        spark_avg = [
            round(
                (sum((o.driver_earning or 0.0) for o in by_day_completed.get(d_, [])) / len(by_day_completed[d_]))
                if by_day_completed.get(d_) else 0.0,
                2,
            )
            for d_ in days_14
        ]

        # Cancel-rate spark: rolling 7-day cancel rate per day.
        cancel_window_rows = [o for o in rows_60 if o.created_at >= cutoff_cancel_spark]
        by_day_all: dict[date, list[M.Order]] = defaultdict(list)
        for o in cancel_window_rows:
            by_day_all[o.created_at.date()].append(o)
        spark_cancel = []
        for d_ in days_14:
            window_lo = d_ - timedelta(days=6)
            bucket: list[M.Order] = []
            for k, vs in by_day_all.items():
                if window_lo <= k <= d_:
                    bucket.extend(vs)
            spark_cancel.append(round(cancel_rate(bucket), 4))

        trust_score, components, _reasons = compute_driver_trust_score(
            rating=d.rating,
            cancel_rate=d.cancel_rate,
            tenure_days=tenure_days,
        )
        breakdown = (
            f"Rating {int(round(components['rating_pts']))} · "
            f"Cancel {int(round(components['cancel_rate_pts']))} · "
            f"Tenure {int(round(components['tenure_pts']))}"
        )

        kpis_list = [
            {
                "id": "earnings_week",
                "label": "This Week Earnings",
                "value": _fmt_currency(earnings_now),
                "value_raw": round(earnings_now, 2),
                "delta_pct": _pct_delta(earnings_now, earnings_prev),
                "fmt": "currency",
                "direction": "higher_is_better",
                "spark": spark_earnings,
            },
            {
                "id": "trips_week",
                "label": "Trips This Week",
                "value": _fmt_int(trips_now),
                "value_raw": trips_now,
                "delta_pct": _pct_delta(trips_now, trips_prev),
                "fmt": "int",
                "direction": "higher_is_better",
                "spark": spark_trips,
            },
            {
                "id": "avg_per_trip",
                "label": "Avg per Trip",
                "value": _fmt_currency(avg_now),
                "value_raw": round(avg_now, 2),
                "delta_pct": _pct_delta(avg_now, avg_prev),
                "fmt": "currency",
                "direction": "higher_is_better",
                "spark": spark_avg,
            },
            {
                "id": "cancel_rate_30d",
                "label": "Cancel Rate (30d)",
                "value": _fmt_percent(cancel_30),
                "value_raw": round(cancel_30, 4),
                "delta_pct": _signed_delta(cancel_30 * 100, cancel_prev_30 * 100),
                "fmt": "percent",
                "direction": "lower_is_better",
                "spark": spark_cancel,
            },
            {
                "id": "trust_score",
                "label": "Trust Score",
                "value": f"{int(round(trust_score))}/100",
                "value_raw": round(trust_score, 1),
                "delta_pct": None,
                "fmt": "int",
                "direction": "higher_is_better",
                "spark": None,
                "breakdown": breakdown,
            },
        ]
        return {"kpis": kpis_list}


# ---------------------------------------------------------------------------
# /earnings-trend
# ---------------------------------------------------------------------------

def _iso_week_start(d: date) -> date:
    """Monday of the ISO week containing d."""
    return d - timedelta(days=d.weekday())


def _month_label(d: date) -> str:
    return d.strftime("%b %Y")


def _short_date(d: date) -> str:
    return d.strftime("%b %d")


def _fetch_peer_orders(s, driver: M.Driver, since: datetime) -> list[M.Order]:
    """Completed orders from peer drivers (same city/vehicle, active, excl self)."""
    peer_ids = [
        r[0] for r in s.execute(
            select(M.Driver.id).where(
                M.Driver.city_id == driver.city_id,
                M.Driver.vehicle_type == driver.vehicle_type,
                M.Driver.is_active == True,  # noqa: E712
                M.Driver.id != driver.id,
            )
        ).all()
    ]
    if not peer_ids:
        return []
    return s.execute(
        select(M.Order).where(
            M.Order.driver_id.in_(peer_ids),
            M.Order.status == "completed",
            M.Order.created_at >= since,
        )
    ).scalars().all()


def _peer_median_per_bucket(
    peer_orders: list[M.Order],
    bucket_keys: list[tuple],
    bucketer,
) -> list[float]:
    """For each bucket, median of (peer driver -> sum earnings in bucket).

    `bucketer(o) -> bucket_key`. Drivers with zero earnings in a bucket are
    excluded from the median to avoid dragging it to zero.
    """
    # bucket_key -> driver_id -> earnings
    sums: dict[tuple, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for o in peer_orders:
        key = bucketer(o)
        if key is None:
            continue
        sums[key][o.driver_id] = sums[key].get(o.driver_id, 0.0) + (o.driver_earning or 0.0)
    out: list[float] = []
    for key in bucket_keys:
        per = [v for v in sums.get(key, {}).values() if v > 0]
        out.append(round(median(per), 2) if per else 0.0)
    return out


@router.get("/earnings-trend")
def earnings_trend(
    range: str = Query(default="W", regex="^[WMY]$"),
    current: CurrentUser = Depends(require_driver),
) -> dict[str, Any]:
    rng = range  # local rename — `range` shadows the builtin in this scope.
    now = datetime.utcnow()
    today = now.date()

    with get_session() as s:
        d = _get_driver_row(s, current)
        joined_date = d.joined_date.date()
        tenure_days = max(0, (now - d.joined_date).days)
        tenure_lbl = _tenure_label(tenure_days)

        # ---------- Bucket the active range ----------
        if rng == "W":
            # 84 daily buckets ending today.
            n_days = 84
            day_keys = [today - timedelta(days=(n_days - 1 - i)) for i in _range(n_days)]
            since = datetime.combine(day_keys[0], datetime.min.time())

            my_orders = s.execute(
                select(M.Order).where(
                    M.Order.driver_id == d.id,
                    M.Order.status == "completed",
                    M.Order.created_at >= since,
                )
            ).scalars().all()
            by_day: dict[date, float] = defaultdict(float)
            for o in my_orders:
                by_day[o.created_at.date()] += (o.driver_earning or 0.0)
            you = [round(by_day.get(k, 0.0), 2) for k in day_keys]

            peer_orders = _fetch_peer_orders(s, d, since)
            peer_median = _peer_median_per_bucket(
                peer_orders,
                [(k,) for k in day_keys],
                lambda o: (o.created_at.date(),),
            )
            buckets = [
                {"start": k.isoformat(), "label": _short_date(k), "you": y, "peer_median": p}
                for k, y, p in zip(day_keys, you, peer_median)
            ]
            months_with_activity = None

        elif rng == "M":
            # 26 weekly buckets (~6 months) ending the current ISO week.
            n_weeks = 26
            current_week_start = _iso_week_start(today)
            week_keys = [current_week_start - timedelta(weeks=(n_weeks - 1 - i)) for i in _range(n_weeks)]
            since = datetime.combine(week_keys[0], datetime.min.time())

            my_orders = s.execute(
                select(M.Order).where(
                    M.Order.driver_id == d.id,
                    M.Order.status == "completed",
                    M.Order.created_at >= since,
                )
            ).scalars().all()
            by_week: dict[date, float] = defaultdict(float)
            for o in my_orders:
                by_week[_iso_week_start(o.created_at.date())] += (o.driver_earning or 0.0)
            you = [round(by_week.get(k, 0.0), 2) for k in week_keys]

            peer_orders = _fetch_peer_orders(s, d, since)
            peer_median = _peer_median_per_bucket(
                peer_orders,
                [(k,) for k in week_keys],
                lambda o: (_iso_week_start(o.created_at.date()),),
            )
            buckets = [
                {"start": k.isoformat(), "label": _short_date(k), "you": y, "peer_median": p}
                for k, y, p in zip(week_keys, you, peer_median)
            ]
            months_with_activity = None

        else:  # Y
            # Monthly buckets back 12 months OR to joined_date, whichever is more recent.
            def _month_floor(dd: date) -> date:
                return date(dd.year, dd.month, 1)

            current_month = _month_floor(today)
            joined_month = _month_floor(joined_date)
            # Walk back at most 12 months but stop at joined_month.
            months: list[date] = []
            cursor = current_month
            for _ in _range(12):
                if cursor < joined_month:
                    break
                months.append(cursor)
                # step back one month
                if cursor.month == 1:
                    cursor = date(cursor.year - 1, 12, 1)
                else:
                    cursor = date(cursor.year, cursor.month - 1, 1)
            months.reverse()
            if not months:
                months = [current_month]
            since = datetime.combine(months[0], datetime.min.time())

            my_orders = s.execute(
                select(M.Order).where(
                    M.Order.driver_id == d.id,
                    M.Order.status == "completed",
                    M.Order.created_at >= since,
                )
            ).scalars().all()
            by_month: dict[date, float] = defaultdict(float)
            for o in my_orders:
                by_month[_month_floor(o.created_at.date())] += (o.driver_earning or 0.0)
            you = [round(by_month.get(k, 0.0), 2) for k in months]
            months_with_activity = sum(1 for v in you if v > 0)

            peer_orders = _fetch_peer_orders(s, d, since)
            peer_median = _peer_median_per_bucket(
                peer_orders,
                [(k,) for k in months],
                lambda o: (_month_floor(o.created_at.date()),),
            )
            buckets = [
                {"start": k.isoformat(), "label": _month_label(k), "you": y, "peer_median": p}
                for k, y, p in zip(months, you, peer_median)
            ]

        # ---------- Best week ever (across all driver history) ----------
        all_completed = s.execute(
            select(M.Order).where(
                M.Order.driver_id == d.id,
                M.Order.status == "completed",
            )
        ).scalars().all()
        best_week_start: date | None = None
        best_amt = 0.0
        if all_completed:
            week_sums: dict[date, float] = defaultdict(float)
            for o in all_completed:
                week_sums[_iso_week_start(o.created_at.date())] += (o.driver_earning or 0.0)
            best_week_start, best_amt = max(week_sums.items(), key=lambda kv: kv[1])

        best_week = (
            {
                "start": best_week_start.isoformat(),
                "label": _short_date(best_week_start),
                "amount": round(best_amt, 2),
            }
            if best_week_start
            else None
        )

        # ---------- Last 30d ----------
        cutoff_30 = now - timedelta(days=30)
        last_30 = [o for o in all_completed if o.created_at >= cutoff_30]
        last_30d = {
            "earnings": round(sum((o.driver_earning or 0.0) for o in last_30), 2),
            "trips": len(last_30),
        }

        return {
            "range": rng,
            "buckets": buckets,
            "best_week": best_week,
            "last_30d": last_30d,
            "tenure_label": tenure_lbl,
            "months_with_activity": months_with_activity,
        }


# ---------------------------------------------------------------------------
# /cancel-trend
# ---------------------------------------------------------------------------

@router.get("/cancel-trend")
def cancel_trend(current: CurrentUser = Depends(require_driver)) -> dict[str, Any]:
    now = datetime.utcnow()
    today = now.date()
    n_weeks = 12
    current_week = _iso_week_start(today)
    # We need 1 prior week of context for the 2-week rolling avg.
    week_keys = [current_week - timedelta(weeks=(n_weeks - 1 - i)) for i in range(n_weeks)]
    earliest_week = week_keys[0] - timedelta(weeks=1)
    since = datetime.combine(earliest_week, datetime.min.time())

    with get_session() as s:
        d = _get_driver_row(s, current)

        # Pull all of MY orders (completed + cancelled) since the rolling window start.
        my_orders = s.execute(
            select(M.Order).where(
                M.Order.driver_id == d.id,
                M.Order.created_at >= since,
            )
        ).scalars().all()
        # Bucket per ISO week.
        my_by_week: dict[date, list[M.Order]] = defaultdict(list)
        for o in my_orders:
            my_by_week[_iso_week_start(o.created_at.date())].append(o)

        # Peer orders (same city, vehicle, active, excl self).
        peer_ids = [
            r[0] for r in s.execute(
                select(M.Driver.id).where(
                    M.Driver.city_id == d.city_id,
                    M.Driver.vehicle_type == d.vehicle_type,
                    M.Driver.is_active == True,  # noqa: E712
                    M.Driver.id != d.id,
                )
            ).all()
        ]
        peer_by_week_driver: dict[date, dict[int, list[M.Order]]] = defaultdict(lambda: defaultdict(list))
        if peer_ids:
            peer_orders = s.execute(
                select(M.Order).where(
                    M.Order.driver_id.in_(peer_ids),
                    M.Order.created_at >= since,
                )
            ).scalars().all()
            for o in peer_orders:
                peer_by_week_driver[_iso_week_start(o.created_at.date())][o.driver_id].append(o)

        def _rate(orders: list[M.Order]) -> float | None:
            cancelled = sum(1 for o in orders if o.status == "cancelled")
            completed = sum(1 for o in orders if o.status == "completed")
            denom = cancelled + completed
            if denom == 0:
                return None
            return cancelled / denom

        buckets: list[dict[str, Any]] = []
        for wk in week_keys:
            prev_wk = wk - timedelta(weeks=1)
            # 2-week rolling for "you"
            window = my_by_week.get(wk, []) + my_by_week.get(prev_wk, [])
            trips_this_week = sum(
                1 for o in my_by_week.get(wk, []) if o.status in ("completed", "cancelled")
            )
            you_rate: float | None
            if trips_this_week < 5:
                you_rate = None
            else:
                you_rate = _rate(window)
                you_rate = round(you_rate, 4) if you_rate is not None else None

            # Peer median: per-driver 2-week rolling cancel rate, then median across drivers.
            peer_rates: list[float] = []
            for did in peer_ids:
                d_window = peer_by_week_driver.get(wk, {}).get(did, []) + \
                           peer_by_week_driver.get(prev_wk, {}).get(did, [])
                # only include peers with at least 5 trips in the window
                d_total = sum(1 for o in d_window if o.status in ("completed", "cancelled"))
                if d_total < 5:
                    continue
                r = _rate(d_window)
                if r is not None:
                    peer_rates.append(r)
            peer_median_val = round(median(peer_rates), 4) if peer_rates else 0.0

            buckets.append({
                "start": wk.isoformat(),
                "label": _short_date(wk),
                "you": you_rate,
                "peer_median": peer_median_val,
                "trips": trips_this_week,
            })

        # Most recent valid (non-null) you/peer for the summary line.
        you_recent: float | None = None
        for b in reversed(buckets):
            if b["you"] is not None:
                you_recent = b["you"]
                break
        peer_recent = buckets[-1]["peer_median"] if buckets else 0.0
        you_vs_peer_pp: float | None
        if you_recent is None:
            you_vs_peer_pp = None
        else:
            you_vs_peer_pp = round((you_recent - peer_recent) * 100, 1)

        return {
            "buckets": buckets,
            "you_recent": you_recent,
            "peer_recent": round(peer_recent, 4),
            "you_vs_peer_pp": you_vs_peer_pp,
        }


# ---------------------------------------------------------------------------
# /peer-comparison
# ---------------------------------------------------------------------------

@router.get("/peer-comparison")
def peer_comparison(current: CurrentUser = Depends(require_driver)) -> dict[str, Any]:
    now = datetime.utcnow()
    cutoff_now = now - timedelta(days=7)

    with get_session() as s:
        d = _get_driver_row(s, current)

        # Vehicle peers (same city + vehicle, active).
        veh_peers = s.execute(
            select(M.Driver).where(
                M.Driver.city_id == d.city_id,
                M.Driver.vehicle_type == d.vehicle_type,
                M.Driver.is_active == True,  # noqa: E712
            )
        ).scalars().all()
        veh_peer_ids = [p.id for p in veh_peers]

        # City peers (same city, any vehicle, active).
        city_peers = s.execute(
            select(M.Driver).where(
                M.Driver.city_id == d.city_id,
                M.Driver.is_active == True,  # noqa: E712
            )
        ).scalars().all()
        city_peer_ids = [p.id for p in city_peers]

        all_relevant_ids = list(set(veh_peer_ids) | set(city_peer_ids))
        orders = s.execute(
            select(M.Order).where(
                M.Order.driver_id.in_(all_relevant_ids),
                M.Order.status == "completed",
                M.Order.created_at >= cutoff_now,
            )
        ).scalars().all()

        sums_per_driver: dict[int, float] = defaultdict(float)
        for o in orders:
            sums_per_driver[o.driver_id] += (o.driver_earning or 0.0)

        # Vehicle-cohort earnings (drivers in same city+vehicle, this week).
        veh_earnings: list[tuple[int, float]] = sorted(
            ((pid, sums_per_driver.get(pid, 0.0)) for pid in veh_peer_ids),
            key=lambda kv: kv[1],
            reverse=True,
        )
        veh_only_amts = [amt for _pid, amt in veh_earnings]

        you_amt = sums_per_driver.get(d.id, 0.0)

        # Median + p90 of the same-cohort peers (excluding the current driver to keep
        # the comparison "you vs them"). If the driver has 0 amt and is not in the
        # cohort, the median already excludes them naturally.
        non_self = [amt for pid, amt in veh_earnings if pid != d.id]

        def _quantile(arr: list[float], q: float) -> float:
            if not arr:
                return 0.0
            arr_sorted = sorted(arr)
            idx = max(0, min(len(arr_sorted) - 1, int(round((len(arr_sorted) - 1) * q))))
            return arr_sorted[idx]

        peer_med = round(_quantile(non_self, 0.5), 2)
        peer_top10 = round(_quantile(non_self, 0.9), 2)

        # Percentile of "you" within the vehicle cohort (incl. self for honest ranking).
        sorted_amts = sorted(veh_only_amts)
        if not sorted_amts:
            percentile = 0
        else:
            below = sum(1 for v in sorted_amts if v < you_amt)
            equal = sum(1 for v in sorted_amts if v == you_amt)
            # mid-rank percentile so identical zeros don't all map to 100.
            percentile = int(round(((below + equal / 2) / len(sorted_amts)) * 100))
        top_pct = max(1, 100 - percentile)
        top_pct_label = f"Top {top_pct}%"

        rank_in_vehicle = next((i + 1 for i, (pid, _amt) in enumerate(veh_earnings) if pid == d.id), len(veh_earnings))

        # Rank in city (across all vehicles in the same city).
        city_earnings = sorted(
            ((pid, sums_per_driver.get(pid, 0.0)) for pid in city_peer_ids),
            key=lambda kv: kv[1],
            reverse=True,
        )
        rank_in_city = next((i + 1 for i, (pid, _amt) in enumerate(city_earnings) if pid == d.id), len(city_earnings))

        return {
            "you_week_earnings": round(you_amt, 2),
            "peer_median": peer_med,
            "peer_top10": peer_top10,
            "percentile": percentile,
            "top_pct_label": top_pct_label,
            "rank_in_vehicle": {"rank": rank_in_vehicle, "of": len(veh_peer_ids)},
            "rank_in_city": {"rank": rank_in_city, "of": len(city_peer_ids)},
            "vehicle_type": d.vehicle_type,
            "city": d.city.name if d.city else "",
        }


# ---------------------------------------------------------------------------
# /incentives
# ---------------------------------------------------------------------------

@router.get("/incentives")
def incentives(current: CurrentUser = Depends(require_driver)) -> dict[str, Any]:
    now = datetime.utcnow()
    with get_session() as s:
        d = _get_driver_row(s, current)
        rows = s.execute(
            select(M.Incentive).where(
                M.Incentive.city_id == d.city_id,
                M.Incentive.starts_at <= now,
                M.Incentive.ends_at >= now,
                M.Incentive.vehicle_type.in_([d.vehicle_type, "any"]),
            ).order_by(M.Incentive.bonus_amount.desc())
        ).scalars().all()
        out: list[dict[str, Any]] = []
        for inc in rows:
            out.append({
                "id": inc.id,
                "title": inc.title,
                "description": inc.description,
                "bonus_amount": round(inc.bonus_amount, 2),
                "zone": inc.zone,
                "zone_label": inc.zone if inc.zone else "City-wide",
                "ends_at": inc.ends_at.isoformat(),
                "ends_in_label": _ends_in_label(now, inc.ends_at),
                "vehicle_type": inc.vehicle_type,
            })
        return {"incentives": out}
