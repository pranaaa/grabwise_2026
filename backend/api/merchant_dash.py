"""Merchant-only personal dashboard endpoints.

Mirrors `backend/api/driver_dash.py` patterns. All endpoints scoped to the
logged-in merchant via `require_merchant`. Peer comparisons are restricted to
other merchants in the same city + cuisine.
"""
from __future__ import annotations
from datetime import datetime, timedelta, date
from collections import defaultdict
from statistics import median, mean, pstdev
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status as http_status
from sqlalchemy import select

_range = range  # preserve the builtin so endpoints can use a `range` query param.

from backend.db.database import get_session
from backend.db import models as M
from backend.api.auth import get_current_user, CurrentUser


router = APIRouter(prefix="/api/merchant", tags=["merchant"])


# ---------------------------------------------------------------------------
# Auth deps
# ---------------------------------------------------------------------------

def require_merchant(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if user.role != "merchant":
        raise HTTPException(status_code=http_status.HTTP_403_FORBIDDEN, detail="merchant access required")
    return user


def _get_merchant_row(s, user: CurrentUser) -> M.Merchant:
    m = s.get(M.Merchant, user.id)
    if not m:
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="merchant record not found")
    return m


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


_PERSONA_LABELS = {
    "lunch-dominant": "Lunch Dominant",
    "dinner-dominant": "Dinner Dominant",
    "weekend-spike": "Weekend Spike",
    "declining-weekends": "Declining Weekends",
    "rising-star": "Rising Star",
}


def _persona_label(persona: str | None) -> str | None:
    if not persona:
        return None
    if persona in _PERSONA_LABELS:
        return _PERSONA_LABELS[persona]
    return " ".join(p.capitalize() for p in persona.replace("_", "-").split("-"))


def _pct_delta(curr: float, prev: float) -> float:
    if prev == 0:
        return 0.0 if curr == 0 else 100.0
    return round(((curr - prev) / prev) * 100, 1)


def _signed_delta(curr: float, prev: float) -> float:
    return round(curr - prev, 1)


def _iso_week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _short_date(d: date) -> str:
    return d.strftime("%b %d")


def _month_label(d: date) -> str:
    return d.strftime("%b %Y")


def _peer_merchant_ids(s, m: M.Merchant) -> list[int]:
    return [
        r[0] for r in s.execute(
            select(M.Merchant.id).where(
                M.Merchant.city_id == m.city_id,
                M.Merchant.cuisine == m.cuisine,
                M.Merchant.id != m.id,
            )
        ).all()
    ]


# ---------------------------------------------------------------------------
# /me
# ---------------------------------------------------------------------------

@router.get("/me")
def me(current: CurrentUser = Depends(require_merchant)) -> dict[str, Any]:
    with get_session() as s:
        m = _get_merchant_row(s, current)
        peer_ids = _peer_merchant_ids(s, m)
        peer_ratings = s.execute(
            select(M.Merchant.rating).where(M.Merchant.id.in_(peer_ids))
        ).scalars().all() if peer_ids else []
        peer_median_rating = round(median(peer_ratings), 2) if peer_ratings else None
        return {
            "id": m.id,
            "name": m.name,
            "city": m.city.name if m.city else "",
            "city_id": m.city_id,
            "zone": m.zone,
            "cuisine": m.cuisine,
            "rating": round(m.rating, 2),
            "avg_prep_min": m.avg_prep_min,
            "behavior_persona": m.behavior_persona,
            "behavior_persona_label": _persona_label(m.behavior_persona),
            "cuisine_peers_in_city": len(peer_ids),
            "cuisine_median_rating_in_city": peer_median_rating,
        }


# ---------------------------------------------------------------------------
# /kpis
# ---------------------------------------------------------------------------

@router.get("/kpis")
def kpis(current: CurrentUser = Depends(require_merchant)) -> dict[str, Any]:
    now = datetime.utcnow()
    cutoff_now = now - timedelta(days=7)
    cutoff_prev_start = now - timedelta(days=14)
    cutoff_30 = now - timedelta(days=30)
    cutoff_prev_30 = now - timedelta(days=60)
    cutoff_spark = now - timedelta(days=14)
    cutoff_cancel_spark = now - timedelta(days=20)

    with get_session() as s:
        m = _get_merchant_row(s, current)

        rows_60 = s.execute(
            select(M.Order).where(
                M.Order.merchant_id == m.id,
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

        gmv_now = sum((o.total or 0.0) for o in completed_now)
        gmv_prev = sum((o.total or 0.0) for o in completed_prev)
        orders_now = len(completed_now)
        orders_prev = len(completed_prev)
        avg_now = (gmv_now / orders_now) if orders_now else 0.0
        avg_prev = (gmv_prev / orders_prev) if orders_prev else 0.0

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
        spark_gmv = [
            round(sum((o.total or 0.0) for o in by_day_completed.get(d_, [])), 2)
            for d_ in days_14
        ]
        spark_orders = [len(by_day_completed.get(d_, [])) for d_ in days_14]
        spark_avg = [
            round(
                (sum((o.total or 0.0) for o in by_day_completed.get(d_, [])) / len(by_day_completed[d_]))
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

        # Rating breakdown vs cuisine peers in the same city.
        peer_ids = _peer_merchant_ids(s, m)
        peer_ratings = s.execute(
            select(M.Merchant.rating).where(M.Merchant.id.in_(peer_ids))
        ).scalars().all() if peer_ids else []
        city_name = m.city.name if m.city else "this city"
        if peer_ratings:
            peer_med = round(median(peer_ratings), 2)
            diff = round(m.rating - peer_med, 2)
            sign = "+" if diff >= 0 else "−"
            breakdown = f"Cuisine median in {city_name}: {peer_med:.2f} (you {sign}{abs(diff):.2f})"
        else:
            breakdown = f"Cuisine median in {city_name}: n/a"

        kpis_list = [
            {
                "id": "gmv",
                "label": "GMV (7d)",
                "value": _fmt_currency(gmv_now),
                "value_raw": round(gmv_now, 2),
                "delta_pct": _pct_delta(gmv_now, gmv_prev),
                "fmt": "currency",
                "direction": "higher_is_better",
                "spark": spark_gmv,
            },
            {
                "id": "orders",
                "label": "Orders (7d)",
                "value": _fmt_int(orders_now),
                "value_raw": orders_now,
                "delta_pct": _pct_delta(orders_now, orders_prev),
                "fmt": "int",
                "direction": "higher_is_better",
                "spark": spark_orders,
            },
            {
                "id": "avg_basket",
                "label": "Avg basket (7d)",
                "value": _fmt_currency(avg_now),
                "value_raw": round(avg_now, 2),
                "delta_pct": _pct_delta(avg_now, avg_prev),
                "fmt": "currency",
                "direction": "higher_is_better",
                "spark": spark_avg,
            },
            {
                "id": "cancel_rate",
                "label": "Cancel rate (30d)",
                "value": _fmt_percent(cancel_30),
                "value_raw": round(cancel_30, 4),
                "delta_pct": _signed_delta(cancel_30 * 100, cancel_prev_30 * 100),
                "fmt": "percent",
                "direction": "lower_is_better",
                "spark": spark_cancel,
            },
            {
                "id": "rating",
                "label": "Rating",
                "value": f"{m.rating:.2f} ★",
                "value_raw": round(m.rating, 2),
                "delta_pct": None,
                "fmt": "rating",
                "direction": "higher_is_better",
                "spark": None,
                "breakdown": breakdown,
            },
        ]
        return {"kpis": kpis_list}


# ---------------------------------------------------------------------------
# /sales-trend
# ---------------------------------------------------------------------------

def _peer_median_per_bucket(
    peer_orders: list[M.Order],
    bucket_keys: list[tuple],
    bucketer,
) -> list[float]:
    """For each bucket, median of (peer merchant -> sum total in bucket).

    Peers with zero in a bucket are excluded from the median to avoid dragging
    it to zero.
    """
    sums: dict[tuple, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for o in peer_orders:
        key = bucketer(o)
        if key is None:
            continue
        sums[key][o.merchant_id] = sums[key].get(o.merchant_id, 0.0) + (o.total or 0.0)
    out: list[float] = []
    for key in bucket_keys:
        per = [v for v in sums.get(key, {}).values() if v > 0]
        out.append(round(median(per), 2) if per else 0.0)
    return out


@router.get("/sales-trend")
def sales_trend(
    range: str = Query(default="W", regex="^[WMY]$"),
    current: CurrentUser = Depends(require_merchant),
) -> dict[str, Any]:
    rng = range
    now = datetime.utcnow()
    today = now.date()

    with get_session() as s:
        m = _get_merchant_row(s, current)
        peer_ids = _peer_merchant_ids(s, m)

        # Earliest order date — used as the floor for the Y range.
        earliest_order_dt = s.execute(
            select(M.Order.created_at).where(
                M.Order.merchant_id == m.id,
                M.Order.status == "completed",
            ).order_by(M.Order.created_at.asc()).limit(1)
        ).scalar_one_or_none()
        earliest_order_date = earliest_order_dt.date() if earliest_order_dt else today

        if rng == "W":
            n_days = 84
            day_keys = [today - timedelta(days=(n_days - 1 - i)) for i in _range(n_days)]
            since = datetime.combine(day_keys[0], datetime.min.time())

            my_orders = s.execute(
                select(M.Order).where(
                    M.Order.merchant_id == m.id,
                    M.Order.status == "completed",
                    M.Order.created_at >= since,
                )
            ).scalars().all()
            by_day: dict[date, float] = defaultdict(float)
            for o in my_orders:
                by_day[o.created_at.date()] += (o.total or 0.0)
            you = [round(by_day.get(k, 0.0), 2) for k in day_keys]

            peer_orders = _fetch_peer_orders(s, peer_ids, since)
            peer_median_vals = _peer_median_per_bucket(
                peer_orders,
                [(k,) for k in day_keys],
                lambda o: (o.created_at.date(),),
            )
            buckets = [
                {"start": k.isoformat(), "label": _short_date(k), "you": y, "peer_median": p}
                for k, y, p in zip(day_keys, you, peer_median_vals)
            ]
            months_with_activity = None

        elif rng == "M":
            n_weeks = 26
            current_week_start = _iso_week_start(today)
            week_keys = [current_week_start - timedelta(weeks=(n_weeks - 1 - i)) for i in _range(n_weeks)]
            since = datetime.combine(week_keys[0], datetime.min.time())

            my_orders = s.execute(
                select(M.Order).where(
                    M.Order.merchant_id == m.id,
                    M.Order.status == "completed",
                    M.Order.created_at >= since,
                )
            ).scalars().all()
            by_week: dict[date, float] = defaultdict(float)
            for o in my_orders:
                by_week[_iso_week_start(o.created_at.date())] += (o.total or 0.0)
            you = [round(by_week.get(k, 0.0), 2) for k in week_keys]

            peer_orders = _fetch_peer_orders(s, peer_ids, since)
            peer_median_vals = _peer_median_per_bucket(
                peer_orders,
                [(k,) for k in week_keys],
                lambda o: (_iso_week_start(o.created_at.date()),),
            )
            buckets = [
                {"start": k.isoformat(), "label": _short_date(k), "you": y, "peer_median": p}
                for k, y, p in zip(week_keys, you, peer_median_vals)
            ]
            months_with_activity = None

        else:  # Y
            def _month_floor(dd: date) -> date:
                return date(dd.year, dd.month, 1)

            current_month = _month_floor(today)
            earliest_month = _month_floor(earliest_order_date)
            months: list[date] = []
            cursor = current_month
            for _ in _range(12):
                if cursor < earliest_month:
                    break
                months.append(cursor)
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
                    M.Order.merchant_id == m.id,
                    M.Order.status == "completed",
                    M.Order.created_at >= since,
                )
            ).scalars().all()
            by_month: dict[date, float] = defaultdict(float)
            for o in my_orders:
                by_month[_month_floor(o.created_at.date())] += (o.total or 0.0)
            you = [round(by_month.get(k, 0.0), 2) for k in months]
            months_with_activity = sum(1 for v in you if v > 0)

            peer_orders = _fetch_peer_orders(s, peer_ids, since)
            peer_median_vals = _peer_median_per_bucket(
                peer_orders,
                [(k,) for k in months],
                lambda o: (_month_floor(o.created_at.date()),),
            )
            buckets = [
                {"start": k.isoformat(), "label": _month_label(k), "you": y, "peer_median": p}
                for k, y, p in zip(months, you, peer_median_vals)
            ]

        # ---------- Best week ever (across all merchant history) ----------
        all_completed = s.execute(
            select(M.Order).where(
                M.Order.merchant_id == m.id,
                M.Order.status == "completed",
            )
        ).scalars().all()
        best_week_start: date | None = None
        best_amt = 0.0
        if all_completed:
            week_sums: dict[date, float] = defaultdict(float)
            for o in all_completed:
                week_sums[_iso_week_start(o.created_at.date())] += (o.total or 0.0)
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

        cutoff_30 = now - timedelta(days=30)
        last_30 = [o for o in all_completed if o.created_at >= cutoff_30]
        last_30d = {
            "gmv": round(sum((o.total or 0.0) for o in last_30), 2),
            "orders": len(last_30),
        }

        if not peer_ids:
            buckets = [{**b, "peer_median": 0.0} for b in buckets]

        return {
            "range": rng,
            "buckets": buckets,
            "best_week": best_week,
            "last_30d": last_30d,
            "peer_count": len(peer_ids),
            "months_with_activity": months_with_activity,
        }


def _fetch_peer_orders(s, peer_ids: list[int], since: datetime) -> list[M.Order]:
    if not peer_ids:
        return []
    return s.execute(
        select(M.Order).where(
            M.Order.merchant_id.in_(peer_ids),
            M.Order.status == "completed",
            M.Order.created_at >= since,
        )
    ).scalars().all()


# ---------------------------------------------------------------------------
# /top-items
# ---------------------------------------------------------------------------

@router.get("/top-items")
def top_items(current: CurrentUser = Depends(require_merchant)) -> dict[str, Any]:
    with get_session() as s:
        m = _get_merchant_row(s, current)
        peer_ids = _peer_merchant_ids(s, m)

        my_items = s.execute(
            select(M.MenuItem).where(M.MenuItem.merchant_id == m.id)
        ).scalars().all()

        peer_items: list[M.MenuItem] = []
        if peer_ids:
            peer_items = s.execute(
                select(M.MenuItem).where(M.MenuItem.merchant_id.in_(peer_ids))
            ).scalars().all()

        # Cuisine median price per item name (across peers; ignore zeros).
        prices_by_name: dict[str, list[float]] = defaultdict(list)
        for it in peer_items:
            if it.price and it.price > 0:
                prices_by_name[it.name.lower()].append(it.price)
        median_price_by_name = {k: round(median(v), 2) for k, v in prices_by_name.items()}

        # Top items.
        top_sorted = sorted(my_items, key=lambda x: (x.popularity or 0), reverse=True)[:8]
        top_out = [
            {
                "id": it.id,
                "name": it.name,
                "price": round(it.price, 2),
                "popularity": int(it.popularity or 0),
                "tags": list(it.tags or []),
            }
            for it in top_sorted
        ]

        # Underperformers: bottom by popularity, but prefer items priced >20% off
        # cuisine median in either direction. Fill with lowest-popularity items
        # if not enough qualify.
        bottom_sorted = sorted(my_items, key=lambda x: (x.popularity or 0))
        candidates: list[tuple[M.MenuItem, float | None, float | None]] = []
        fallback: list[tuple[M.MenuItem, float | None, float | None]] = []
        for it in bottom_sorted:
            med_price = median_price_by_name.get(it.name.lower())
            if med_price and med_price > 0:
                diff_pct = round(((it.price - med_price) / med_price) * 100, 1)
            else:
                diff_pct = None
            tup = (it, med_price, diff_pct)
            if diff_pct is not None and abs(diff_pct) > 20:
                candidates.append(tup)
            else:
                fallback.append(tup)

        chosen = candidates[:3]
        if len(chosen) < 3:
            chosen = chosen + fallback[: 3 - len(chosen)]
        chosen = chosen[:3]

        underperformers = [
            {
                "id": it.id,
                "name": it.name,
                "price": round(it.price, 2),
                "popularity": int(it.popularity or 0),
                "tags": list(it.tags or []),
                "cuisine_median_price": med_price,
                "price_diff_pct": diff_pct,
            }
            for (it, med_price, diff_pct) in chosen
        ]
        return {"top_items": top_out, "underperformers": underperformers}


# ---------------------------------------------------------------------------
# /competitor-signals
# ---------------------------------------------------------------------------

@router.get("/competitor-signals")
def competitor_signals(current: CurrentUser = Depends(require_merchant)) -> dict[str, Any]:
    now = datetime.utcnow()
    cutoff_30 = now - timedelta(days=30)

    with get_session() as s:
        m = _get_merchant_row(s, current)
        peer_ids = _peer_merchant_ids(s, m)
        city_name = m.city.name if m.city else ""

        peer_rows = s.execute(
            select(M.Merchant).where(M.Merchant.id.in_(peer_ids))
        ).scalars().all() if peer_ids else []

        # Rating
        peer_ratings = [p.rating for p in peer_rows]
        rating_med = round(median(peer_ratings), 2) if peer_ratings else m.rating
        rating_block = {
            "you": round(m.rating, 2),
            "peer_median": rating_med,
            "delta": round(m.rating - rating_med, 2),
            "direction": "higher_is_better",
        }

        # Avg prep time (lower is better)
        peer_prep = [p.avg_prep_min for p in peer_rows]
        prep_med = int(round(median(peer_prep))) if peer_prep else m.avg_prep_min
        prep_block = {
            "you": m.avg_prep_min,
            "peer_median": prep_med,
            "delta": m.avg_prep_min - prep_med,
            "direction": "lower_is_better",
        }

        # Avg item price (neutral)
        my_items = s.execute(
            select(M.MenuItem).where(M.MenuItem.merchant_id == m.id)
        ).scalars().all()
        my_prices = [it.price for it in my_items if it.price and it.price > 0]
        my_avg_price = round(mean(my_prices), 2) if my_prices else 0.0

        peer_prices: list[float] = []
        if peer_ids:
            peer_prices = [
                p for (p,) in s.execute(
                    select(M.MenuItem.price).where(M.MenuItem.merchant_id.in_(peer_ids))
                ).all()
                if p and p > 0
            ]
        peer_price_med = round(median(peer_prices), 2) if peer_prices else my_avg_price
        price_block = {
            "you": my_avg_price,
            "peer_median": peer_price_med,
            "delta": round(my_avg_price - peer_price_med, 2),
            "direction": "neutral",
        }

        # Order volume 30d (higher is better)
        my_30 = s.execute(
            select(M.Order).where(
                M.Order.merchant_id == m.id,
                M.Order.status == "completed",
                M.Order.created_at >= cutoff_30,
            )
        ).scalars().all()
        my_volume = len(my_30)

        peer_volume_med = 0
        if peer_ids:
            peer_orders = s.execute(
                select(M.Order).where(
                    M.Order.merchant_id.in_(peer_ids),
                    M.Order.status == "completed",
                    M.Order.created_at >= cutoff_30,
                )
            ).scalars().all()
            counts: dict[int, int] = defaultdict(int)
            for o in peer_orders:
                counts[o.merchant_id] += 1
            # Include peers with zero in the denominator.
            per_peer = [counts.get(pid, 0) for pid in peer_ids]
            peer_volume_med = int(round(median(per_peer))) if per_peer else 0
        volume_block = {
            "you": my_volume,
            "peer_median": peer_volume_med,
            "delta": my_volume - peer_volume_med,
            "direction": "higher_is_better",
        }

        return {
            "rating": rating_block,
            "prep_time": prep_block,
            "avg_item_price": price_block,
            "order_volume_30d": volume_block,
            "peer_count": len(peer_ids),
            "cuisine": m.cuisine,
            "city": city_name,
        }


# ---------------------------------------------------------------------------
# /demand-forecast
# ---------------------------------------------------------------------------

_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_DAY_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _confidence(stdv: float, mn: float) -> str:
    if mn <= 0:
        return "low"
    cv = stdv / mn
    if cv < 0.4:
        return "high"
    if cv < 0.8:
        return "med"
    return "low"


def _format_peak_label(d: date, hour: int) -> str:
    name = _DAY_NAMES[d.weekday()]
    return f"{name} · {hour:02d}:00–{(hour + 1) % 24:02d}:00"


@router.get("/demand-forecast")
def demand_forecast(current: CurrentUser = Depends(require_merchant)) -> dict[str, Any]:
    now = datetime.utcnow()
    today = now.date()
    history_window = 28
    cutoff = now - timedelta(days=history_window)

    with get_session() as s:
        m = _get_merchant_row(s, current)
        zone = m.zone

        rows = s.execute(
            select(M.Order).where(
                M.Order.merchant_id == m.id,
                M.Order.status == "completed",
                M.Order.created_at >= cutoff,
            )
        ).scalars().all()

        # If we have fewer than 14 distinct days of activity, mark as not-enough.
        distinct_days = {o.created_at.date() for o in rows}
        if len(distinct_days) < 14:
            return {
                "enough_history": False,
                "days": [],
                "peak_windows": [],
                "busiest_peak": None,
                "history_window_days": history_window,
            }

        # Bucket per (date, dow, hour) -> count for that exact date+hour. Then
        # groupings by (dow, hour) across the 28-day window give us 4 samples.
        per_date_hour: dict[tuple[date, int], int] = defaultdict(int)
        for o in rows:
            dt = o.created_at
            per_date_hour[(dt.date(), dt.hour)] += 1

        slot_samples: dict[tuple[int, int], list[int]] = defaultdict(list)
        # We want 4 samples per (dow, hour) — one per matching day in the window.
        # Walk all dates in [cutoff_date, today-1]; for each (date, hour), record.
        cutoff_date = cutoff.date()
        d_iter = cutoff_date
        while d_iter <= today:
            for h in range(24):
                slot_samples[(d_iter.weekday(), h)].append(per_date_hour.get((d_iter, h), 0))
            d_iter = d_iter + timedelta(days=1)

        # Predict for next 7 days starting tomorrow.
        future_days: list[date] = [today + timedelta(days=i) for i in range(1, 8)]
        days_out: list[dict[str, Any]] = []
        # collect (date, hour, predicted, confidence) for top windows.
        per_day_hour_preds: list[tuple[date, int, int, str]] = []

        for d_ in future_days:
            dow = d_.weekday()
            day_total = 0
            best_hour = None
            best_pred = -1
            best_conf = "low"
            for h in range(24):
                samples = slot_samples.get((dow, h), [])
                if not samples:
                    pred = 0
                    conf = "low"
                else:
                    mn = mean(samples)
                    stdv = pstdev(samples) if len(samples) > 1 else 0.0
                    pred = int(round(mn))
                    conf = _confidence(stdv, mn)
                day_total += pred
                if pred > best_pred:
                    best_pred = pred
                    best_hour = h
                    best_conf = conf
            # Determine day-level confidence by best hour's confidence.
            days_out.append({
                "date": d_.isoformat(),
                "label": _DAY_SHORT[dow],
                "predicted_orders": day_total,
                "confidence": best_conf,
            })
            if best_hour is not None and best_pred > 0:
                per_day_hour_preds.append((d_, best_hour, best_pred, best_conf))

        # Top 3 peaks: one per day max, then take 3 highest.
        per_day_hour_preds.sort(key=lambda t: t[2], reverse=True)
        top3 = per_day_hour_preds[:3]
        peak_windows = [
            {
                "date": d_.isoformat(),
                "label": _format_peak_label(d_, h),
                "hour": h,
                "predicted_orders": pred,
                "confidence": conf,
                "zone": zone,
            }
            for (d_, h, pred, conf) in top3
        ]

        busiest = None
        if peak_windows:
            top = top3[0]
            d_, h, pred, conf = top
            busiest = {
                "date": d_.isoformat(),
                "label": f"{_DAY_NAMES[d_.weekday()]} · {h:02d}:00",
                "hour": h,
                "zone": zone,
                "predicted_orders": pred,
            }

        return {
            "enough_history": True,
            "days": days_out,
            "peak_windows": peak_windows,
            "busiest_peak": busiest,
            "history_window_days": history_window,
        }
