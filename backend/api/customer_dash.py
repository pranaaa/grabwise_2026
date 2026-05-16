"""Customer-only personal dashboard endpoints.

Mirrors `backend/api/merchant_dash.py` patterns. All endpoints scoped to the
logged-in customer via `require_customer`. Peer comparisons are restricted to
other customers in the same city.
"""
from __future__ import annotations
from datetime import datetime, timedelta, date
from collections import defaultdict
from statistics import median, mean
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status as http_status
from sqlalchemy import select

_range = range  # preserve the builtin so endpoints can use a `range` query param.

from backend.db.database import get_session
from backend.db import models as M
from backend.api.auth import get_current_user, CurrentUser
from backend.tools._risk_math import compute_driver_trust_score


router = APIRouter(prefix="/api/customer", tags=["customer"])


# ---------------------------------------------------------------------------
# Auth deps
# ---------------------------------------------------------------------------

def require_customer(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if user.role != "customer":
        raise HTTPException(status_code=http_status.HTTP_403_FORBIDDEN, detail="customer access required")
    return user


def _get_customer_row(s, user: CurrentUser) -> M.Customer:
    c = s.get(M.Customer, user.id)
    if not c:
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="customer record not found")
    return c


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


def _tenure_label(days: int) -> str:
    days = max(0, int(days))
    years = days // 365
    months = (days % 365) // 30
    if years <= 0 and months <= 0:
        return f"{days}d"
    if years <= 0:
        return f"{months}m"
    return f"{years}y {months}m"


_PERSONA_LABELS = {
    "late-night-orderer": "Late-Night Orderer",
    "lunch-regular": "Lunch Regular",
    "weekend-foodie": "Weekend Foodie",
    "vegetarian-explorer": "Vegetarian Explorer",
    "new-user": "New User",
    "high-spender": "High Spender",
}


def _persona_label(persona: str | None) -> str | None:
    if not persona:
        return None
    if persona in _PERSONA_LABELS:
        return _PERSONA_LABELS[persona]
    return " ".join(p.capitalize() for p in persona.replace("_", "-").split("-"))


_DIETARY_LABELS = {
    "vegetarian": "Vegetarian",
    "vegan": "Vegan",
    "halal": "Halal",
    "gluten-free": "Gluten-Free",
    "kosher": "Kosher",
}


def _dietary_label(tag: str) -> str:
    if tag in _DIETARY_LABELS:
        return _DIETARY_LABELS[tag]
    return " ".join(p.capitalize() for p in tag.replace("_", "-").split("-"))


def _pct_delta(curr: float, prev: float) -> float:
    if prev == 0:
        return 0.0 if curr == 0 else 100.0
    return round(((curr - prev) / prev) * 100, 1)


def _iso_week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _short_date(d: date) -> str:
    return d.strftime("%b %d")


def _month_label(d: date) -> str:
    return d.strftime("%b %Y")


def _is_late_night(dt: datetime) -> bool:
    h = dt.hour
    return h >= 22 or h < 5


def _peer_customer_ids(s, c: M.Customer) -> list[int]:
    return [
        r[0] for r in s.execute(
            select(M.Customer.id).where(
                M.Customer.city_id == c.city_id,
                M.Customer.id != c.id,
            )
        ).all()
    ]


def _relative_label(now: datetime, dt: datetime) -> str:
    delta = now - dt
    secs = delta.total_seconds()
    if secs < 0:
        return "just now"
    mins = int(secs // 60)
    if mins < 1:
        return "just now"
    if mins < 60:
        return f"{mins}m ago"
    hrs = mins // 60
    if hrs < 24:
        return f"{hrs}h ago"
    days = hrs // 24
    if days <= 6:
        return f"{days}d ago"
    return dt.strftime("%b %d")


# ---------------------------------------------------------------------------
# /me
# ---------------------------------------------------------------------------

@router.get("/me")
def me(current: CurrentUser = Depends(require_customer)) -> dict[str, Any]:
    now = datetime.utcnow()
    with get_session() as s:
        c = _get_customer_row(s, current)
        tenure_days = max(0, (now - c.signup_date).days)
        peer_ids = _peer_customer_ids(s, c)
        prefs = list(c.dietary_prefs or [])
        return {
            "id": c.id,
            "name": c.name,
            "city": c.city.name if c.city else "",
            "city_id": c.city_id,
            "dietary_prefs": prefs,
            "dietary_prefs_labels": [_dietary_label(p) for p in prefs],
            "signup_date": c.signup_date.isoformat(),
            "tenure_days": tenure_days,
            "tenure_label": _tenure_label(tenure_days),
            "behavior_persona": c.behavior_persona,
            "behavior_persona_label": _persona_label(c.behavior_persona),
            "city_peer_count": len(peer_ids),
        }


# ---------------------------------------------------------------------------
# /kpis
# ---------------------------------------------------------------------------

@router.get("/kpis")
def kpis(current: CurrentUser = Depends(require_customer)) -> dict[str, Any]:
    now = datetime.utcnow()
    cutoff_30 = now - timedelta(days=30)
    cutoff_prev_30 = now - timedelta(days=60)
    cutoff_spark = now - timedelta(days=14)

    with get_session() as s:
        c = _get_customer_row(s, current)

        rows_60 = s.execute(
            select(M.Order).where(
                M.Order.customer_id == c.id,
                M.Order.created_at >= cutoff_prev_30,
            )
        ).scalars().all()

        completed_now = [o for o in rows_60 if o.status == "completed" and o.created_at >= cutoff_30]
        completed_prev = [
            o for o in rows_60
            if o.status == "completed" and cutoff_prev_30 <= o.created_at < cutoff_30
        ]

        spend_now = sum((o.total or 0.0) for o in completed_now)
        spend_prev = sum((o.total or 0.0) for o in completed_prev)
        orders_now = len(completed_now)
        orders_prev = len(completed_prev)
        avg_now = (spend_now / orders_now) if orders_now else 0.0
        avg_prev = (spend_prev / orders_prev) if orders_prev else 0.0

        days_14 = [(now - timedelta(days=i)).date() for i in range(13, -1, -1)]
        spark_rows = [o for o in rows_60 if o.created_at >= cutoff_spark and o.status == "completed"]
        by_day_completed: dict[date, list[M.Order]] = defaultdict(list)
        for o in spark_rows:
            by_day_completed[o.created_at.date()].append(o)
        spark_orders = [len(by_day_completed.get(d_, [])) for d_ in days_14]
        spark_spend = [
            round(sum((o.total or 0.0) for o in by_day_completed.get(d_, [])), 2)
            for d_ in days_14
        ]
        spark_avg = [
            round(
                (sum((o.total or 0.0) for o in by_day_completed.get(d_, [])) / len(by_day_completed[d_]))
                if by_day_completed.get(d_) else 0.0,
                2,
            )
            for d_ in days_14
        ]

        # Top cuisine — lifetime, by count.
        all_completed_lifetime = s.execute(
            select(M.Order).where(
                M.Order.customer_id == c.id,
                M.Order.status == "completed",
            )
        ).scalars().all()
        merchant_ids_lifetime = {o.merchant_id for o in all_completed_lifetime}
        cuisine_by_merchant: dict[int, str] = {}
        if merchant_ids_lifetime:
            mer_rows = s.execute(
                select(M.Merchant.id, M.Merchant.cuisine).where(
                    M.Merchant.id.in_(merchant_ids_lifetime)
                )
            ).all()
            cuisine_by_merchant = {mid: cu for (mid, cu) in mer_rows}
        cuisine_counts: dict[str, int] = defaultdict(int)
        cuisine_merchants: dict[str, set] = defaultdict(set)
        for o in all_completed_lifetime:
            cu = cuisine_by_merchant.get(o.merchant_id)
            if cu:
                cuisine_counts[cu] += 1
                cuisine_merchants[cu].add(o.merchant_id)
        if cuisine_counts:
            top_cuisine = max(cuisine_counts.items(), key=lambda kv: kv[1])[0]
            top_cuisine_breakdown = (
                f"{cuisine_counts[top_cuisine]} orders · "
                f"{len(cuisine_merchants[top_cuisine])} spots"
            )
        else:
            top_cuisine = "—"
            top_cuisine_breakdown = "No orders yet"

        # Late-night orders (30d) + average driver trust.
        late_night_orders = [o for o in completed_now if _is_late_night(o.created_at)]
        late_night_count = len(late_night_orders)
        late_night_breakdown = "No late-night orders yet"
        if late_night_count:
            driver_ids = [o.driver_id for o in late_night_orders if o.driver_id is not None]
            trust_scores: list[float] = []
            if driver_ids:
                drivers = s.execute(
                    select(M.Driver).where(M.Driver.id.in_(driver_ids))
                ).scalars().all()
                drv_by_id = {d.id: d for d in drivers}
                for did in driver_ids:
                    d = drv_by_id.get(did)
                    if not d:
                        continue
                    tenure = max(0, (now - d.joined_date).days)
                    score, _comp, _r = compute_driver_trust_score(
                        rating=d.rating,
                        cancel_rate=d.cancel_rate,
                        tenure_days=tenure,
                    )
                    trust_scores.append(score)
            if trust_scores:
                avg_trust = int(round(sum(trust_scores) / len(trust_scores)))
                late_night_breakdown = f"Avg driver trust on these: {avg_trust}/100"
            else:
                late_night_breakdown = "Drivers not assigned to score"

        kpis_list: list[dict[str, Any]] = [
            {
                "id": "orders",
                "label": "Orders (30d)",
                "value": _fmt_int(orders_now),
                "value_raw": orders_now,
                "delta_pct": _pct_delta(orders_now, orders_prev),
                "fmt": "int",
                "direction": "higher_is_better",
                "spark": spark_orders,
            },
            {
                "id": "spend",
                "label": "Spend (30d)",
                "value": _fmt_currency(spend_now),
                "value_raw": round(spend_now, 2),
                "delta_pct": _pct_delta(spend_now, spend_prev),
                "fmt": "currency",
                "direction": "neutral",
                "spark": spark_spend,
            },
            {
                "id": "avg_basket",
                "label": "Avg basket (30d)",
                "value": _fmt_currency(avg_now),
                "value_raw": round(avg_now, 2),
                "delta_pct": _pct_delta(avg_now, avg_prev),
                "fmt": "currency",
                "direction": "higher_is_better",
                "spark": spark_avg,
            },
            {
                "id": "top_cuisine",
                "label": "Top cuisine",
                "value": top_cuisine,
                "delta_pct": None,
                "fmt": "text",
                "direction": "neutral",
                "spark": None,
                "breakdown": top_cuisine_breakdown,
            },
            {
                "id": "late_night_orders",
                "label": "Late-night orders (30d)",
                "value": _fmt_int(late_night_count),
                "value_raw": late_night_count,
                "delta_pct": None,
                "fmt": "int",
                "direction": "neutral",
                "spark": None,
                "breakdown": late_night_breakdown,
            },
        ]
        return {"kpis": kpis_list}


# ---------------------------------------------------------------------------
# /spend-trend
# ---------------------------------------------------------------------------

def _peer_median_per_bucket(
    peer_orders: list[M.Order],
    bucket_keys: list[tuple],
    bucketer,
) -> list[float]:
    """For each bucket, median of (peer customer -> sum total in bucket).

    Peers with zero in a bucket are excluded from the median to avoid dragging
    it to zero.
    """
    sums: dict[tuple, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for o in peer_orders:
        key = bucketer(o)
        if key is None:
            continue
        sums[key][o.customer_id] = sums[key].get(o.customer_id, 0.0) + (o.total or 0.0)
    out: list[float] = []
    for key in bucket_keys:
        per = [v for v in sums.get(key, {}).values() if v > 0]
        out.append(round(median(per), 2) if per else 0.0)
    return out


def _fetch_peer_orders(s, peer_ids: list[int], since: datetime) -> list[M.Order]:
    if not peer_ids:
        return []
    return s.execute(
        select(M.Order).where(
            M.Order.customer_id.in_(peer_ids),
            M.Order.status == "completed",
            M.Order.created_at >= since,
        )
    ).scalars().all()


@router.get("/spend-trend")
def spend_trend(
    range: str = Query(default="W", regex="^[WMY]$"),
    current: CurrentUser = Depends(require_customer),
) -> dict[str, Any]:
    rng = range
    now = datetime.utcnow()
    today = now.date()

    with get_session() as s:
        c = _get_customer_row(s, current)
        peer_ids = _peer_customer_ids(s, c)
        signup_date = c.signup_date.date()
        tenure_days = max(0, (now - c.signup_date).days)
        tenure_lbl = _tenure_label(tenure_days)

        if rng == "W":
            n_days = 84
            day_keys = [today - timedelta(days=(n_days - 1 - i)) for i in _range(n_days)]
            since = datetime.combine(day_keys[0], datetime.min.time())

            my_orders = s.execute(
                select(M.Order).where(
                    M.Order.customer_id == c.id,
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
                    M.Order.customer_id == c.id,
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
            signup_month = _month_floor(signup_date)
            months: list[date] = []
            cursor = current_month
            for _ in _range(12):
                if cursor < signup_month:
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
                    M.Order.customer_id == c.id,
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

        # ---------- Best week so far (across all customer history) ----------
        all_completed = s.execute(
            select(M.Order).where(
                M.Order.customer_id == c.id,
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
            "spend": round(sum((o.total or 0.0) for o in last_30), 2),
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
            "tenure_label": tenure_lbl,
            "months_with_activity": months_with_activity,
        }


# ---------------------------------------------------------------------------
# /recent-orders
# ---------------------------------------------------------------------------

@router.get("/recent-orders")
def recent_orders(
    limit: int = Query(default=10, ge=1, le=50),
    current: CurrentUser = Depends(require_customer),
) -> dict[str, Any]:
    now = datetime.utcnow()
    with get_session() as s:
        c = _get_customer_row(s, current)
        rows = s.execute(
            select(M.Order).where(
                M.Order.customer_id == c.id,
            ).order_by(M.Order.created_at.desc()).limit(limit)
        ).scalars().all()

        merchant_ids = {o.merchant_id for o in rows}
        merchants_by_id: dict[int, M.Merchant] = {}
        if merchant_ids:
            mer_rows = s.execute(
                select(M.Merchant).where(M.Merchant.id.in_(merchant_ids))
            ).scalars().all()
            merchants_by_id = {m.id: m for m in mer_rows}

        out = []
        for o in rows:
            m = merchants_by_id.get(o.merchant_id)
            out.append({
                "id": o.id,
                "created_at": o.created_at.isoformat(),
                "label": _relative_label(now, o.created_at),
                "merchant_id": o.merchant_id,
                "merchant_name": m.name if m else "—",
                "merchant_cuisine": m.cuisine if m else "",
                "merchant_rating": round(m.rating, 2) if m else None,
                "total": round(o.total or 0.0, 2),
                "status": o.status,
                "pickup_zone": o.pickup_zone,
                "dropoff_zone": o.dropoff_zone,
                "late_night": _is_late_night(o.created_at),
            })
        return {"orders": out}


# ---------------------------------------------------------------------------
# /favorites
# ---------------------------------------------------------------------------

@router.get("/favorites")
def favorites(current: CurrentUser = Depends(require_customer)) -> dict[str, Any]:
    with get_session() as s:
        c = _get_customer_row(s, current)
        all_completed = s.execute(
            select(M.Order).where(
                M.Order.customer_id == c.id,
                M.Order.status == "completed",
            )
        ).scalars().all()

        total_orders = len(all_completed)
        if total_orders == 0:
            return {"top_merchants": [], "cuisine_breakdown": [], "total_orders": 0}

        merchant_ids = {o.merchant_id for o in all_completed}
        mer_rows = s.execute(
            select(M.Merchant).where(M.Merchant.id.in_(merchant_ids))
        ).scalars().all()
        merchants_by_id = {m.id: m for m in mer_rows}

        # Top merchants by order count.
        per_mer_count: dict[int, int] = defaultdict(int)
        per_mer_spend: dict[int, float] = defaultdict(float)
        for o in all_completed:
            per_mer_count[o.merchant_id] += 1
            per_mer_spend[o.merchant_id] += (o.total or 0.0)

        top_mer_sorted = sorted(
            per_mer_count.items(),
            key=lambda kv: (kv[1], per_mer_spend[kv[0]]),
            reverse=True,
        )[:5]
        top_merchants = []
        for mid, cnt in top_mer_sorted:
            m = merchants_by_id.get(mid)
            if not m:
                continue
            top_merchants.append({
                "id": mid,
                "name": m.name,
                "cuisine": m.cuisine,
                "rating": round(m.rating, 2),
                "order_count": cnt,
                "total_spent": round(per_mer_spend[mid], 2),
                "city": m.city.name if m.city else "",
            })

        # Cuisine breakdown.
        cuisine_count: dict[str, int] = defaultdict(int)
        cuisine_total: dict[str, float] = defaultdict(float)
        for o in all_completed:
            m = merchants_by_id.get(o.merchant_id)
            if not m or not m.cuisine:
                continue
            cuisine_count[m.cuisine] += 1
            cuisine_total[m.cuisine] += (o.total or 0.0)

        cuisine_sorted = sorted(
            cuisine_count.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:5]
        cuisine_breakdown = [
            {"cuisine": cu, "count": cnt, "total": round(cuisine_total[cu], 2)}
            for cu, cnt in cuisine_sorted
        ]

        return {
            "top_merchants": top_merchants,
            "cuisine_breakdown": cuisine_breakdown,
            "total_orders": total_orders,
        }


# ---------------------------------------------------------------------------
# /smart-picks
# ---------------------------------------------------------------------------

@router.get("/smart-picks")
def smart_picks(current: CurrentUser = Depends(require_customer)) -> dict[str, Any]:
    now = datetime.utcnow()
    cutoff_14 = now - timedelta(days=14)

    with get_session() as s:
        c = _get_customer_row(s, current)
        prefs = list(c.dietary_prefs or [])
        prefs_set = {p.lower() for p in prefs}

        # Merchants the customer ordered from in the last 14 days — exclude.
        recent_merchant_ids = {
            r[0] for r in s.execute(
                select(M.Order.merchant_id).where(
                    M.Order.customer_id == c.id,
                    M.Order.created_at >= cutoff_14,
                )
            ).all()
        }

        # All merchants in the customer's city.
        city_merchants = s.execute(
            select(M.Merchant).where(M.Merchant.city_id == c.city_id)
        ).scalars().all()

        if not city_merchants:
            return {"picks": [], "dietary_filter": ",".join(prefs), "selection_strategy": "top_rated"}

        merchant_ids = [m.id for m in city_merchants]
        all_items = s.execute(
            select(M.MenuItem).where(M.MenuItem.merchant_id.in_(merchant_ids))
        ).scalars().all()
        items_by_merchant: dict[int, list[M.MenuItem]] = defaultdict(list)
        for it in all_items:
            items_by_merchant[it.merchant_id].append(it)

        def _matches_prefs(it: M.MenuItem) -> bool:
            if not prefs_set:
                return False
            tags = {t.lower() for t in (it.tags or [])}
            return bool(tags & prefs_set)

        # Step 1-3: dietary-matched candidates.
        dietary_candidates: list[tuple[M.Merchant, list[M.MenuItem], int]] = []
        for m in city_merchants:
            if m.id in recent_merchant_ids:
                continue
            items = items_by_merchant.get(m.id, [])
            if prefs_set:
                matched = [it for it in items if _matches_prefs(it)]
                if len(matched) < 2:
                    continue
                pop_sum = sum((it.popularity or 0) for it in matched)
                dietary_candidates.append((m, matched, pop_sum))

        dietary_candidates.sort(
            key=lambda t: (t[0].rating, t[2]),
            reverse=True,
        )

        picks: list[dict[str, Any]] = []
        chosen_ids: set[int] = set()

        def _build_pick(m: M.Merchant, matched_items: list[M.MenuItem]) -> dict[str, Any]:
            sorted_items = sorted(matched_items, key=lambda it: (it.popularity or 0), reverse=True)[:2]
            return {
                "merchant_id": m.id,
                "name": m.name,
                "cuisine": m.cuisine,
                "rating": round(m.rating, 2),
                "avg_prep_min": m.avg_prep_min,
                "zone": m.zone,
                "city": m.city.name if m.city else "",
                "matched_items": [
                    {
                        "id": it.id,
                        "name": it.name,
                        "price": round(it.price, 2),
                        "tags": list(it.tags or []),
                    }
                    for it in sorted_items
                ],
            }

        for (m, matched, _pop) in dietary_candidates:
            if len(picks) >= 4:
                break
            picks.append(_build_pick(m, matched))
            chosen_ids.add(m.id)

        dietary_used = len(picks)
        # Fallback fill: top-rated merchants in city, excluding ones already chosen + recent.
        if len(picks) < 4:
            fallback_pool = [
                m for m in city_merchants
                if m.id not in chosen_ids and m.id not in recent_merchant_ids
            ]
            fallback_pool.sort(key=lambda m: (m.rating, m.avg_prep_min and -m.avg_prep_min or 0), reverse=True)
            for m in fallback_pool:
                if len(picks) >= 4:
                    break
                items = items_by_merchant.get(m.id, [])
                if prefs_set:
                    matched = [it for it in items if _matches_prefs(it)]
                    if len(matched) < 2:
                        # No dietary-matching items — pick top popular items as best-effort.
                        matched = sorted(items, key=lambda it: (it.popularity or 0), reverse=True)[:2]
                else:
                    matched = sorted(items, key=lambda it: (it.popularity or 0), reverse=True)[:2]
                picks.append(_build_pick(m, matched))
                chosen_ids.add(m.id)

        # Final fallback: if we somehow still have <4 (e.g. recent_merchant_ids drained the pool),
        # backfill from recent_merchant_ids by rating.
        if len(picks) < 4:
            backfill = [m for m in city_merchants if m.id not in chosen_ids]
            backfill.sort(key=lambda m: m.rating, reverse=True)
            for m in backfill:
                if len(picks) >= 4:
                    break
                items = items_by_merchant.get(m.id, [])
                if prefs_set:
                    matched = [it for it in items if _matches_prefs(it)]
                    if len(matched) < 2:
                        matched = sorted(items, key=lambda it: (it.popularity or 0), reverse=True)[:2]
                else:
                    matched = sorted(items, key=lambda it: (it.popularity or 0), reverse=True)[:2]
                picks.append(_build_pick(m, matched))
                chosen_ids.add(m.id)

        if not prefs_set:
            strategy = "top_rated"
        elif dietary_used == len(picks) and len(picks) > 0:
            strategy = "dietary_match"
        elif dietary_used > 0:
            strategy = "hybrid"
        else:
            strategy = "top_rated"

        return {
            "picks": picks[:4],
            "dietary_filter": ",".join(prefs),
            "selection_strategy": strategy,
        }
