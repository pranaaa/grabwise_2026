"""Tools the Customer Convenience Agent calls.

Pillars from the deck:
  • Smart Discovery — items based on user preferences, dietary needs, allergies, and location.
  • Safe Late-Night Matching — prioritize highly rated and reliable drivers for late-night orders.

Tier-1 customer refinements (May 2026):
  • get_typical_pattern — pre-computed customer anchor (typical hour, basket, recency, loyalty)
  • get_customer_profile now surfaces behavior_persona, tenure_tier, total_orders, lifetime_spend
  • get_customer_recent_orders now reports cuisine_drift + last_order_days_ago
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timedelta
from statistics import median
from typing import Any

from sqlalchemy import select, and_, func
from langchain_core.tools import tool

from backend.db.database import get_session
from backend.db import models as M


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tenure_tier(signup_date: datetime) -> str:
    days = (datetime.utcnow() - signup_date).days
    if days < 30:
        return "new"        # < 1 month
    if days < 180:
        return "regular"    # 1–6 months
    return "loyal"          # 6+ months


def _loyalty_tier(total_orders: int) -> str:
    if total_orders < 5:
        return "low"
    if total_orders < 25:
        return "medium"
    return "high"


def _hour_bucket(h: int) -> str:
    if 5 <= h < 11:  return "morning"
    if 11 <= h < 14: return "lunch"
    if 14 <= h < 17: return "afternoon"
    if 17 <= h < 22: return "dinner"
    return "late-night"


# ---------------------------------------------------------------------------
# 1. Customer profile — enriched
# ---------------------------------------------------------------------------
@tool
def get_customer_profile(customer_id: int) -> dict[str, Any]:
    """Look up a customer's profile: name, city, dietary preferences, tenure,
    behavior persona, total orders, and lifetime spend.

    Args:
        customer_id: The customer's numeric ID.
    """
    with get_session() as s:
        c: M.Customer | None = s.get(M.Customer, customer_id)
        if not c:
            return {"error": f"customer {customer_id} not found"}

        agg = s.execute(
            select(
                func.count(M.Order.id),
                func.coalesce(func.sum(M.Order.total), 0.0),
            )
            .where(
                M.Order.customer_id == customer_id,
                M.Order.status == "completed",
            )
        ).one()
        total_orders = int(agg[0] or 0)
        lifetime_spend = round(float(agg[1] or 0.0), 2)

        return {
            "customer_id": c.id,
            "name": c.name,
            "city": c.city.name,
            "dietary_prefs": c.dietary_prefs or [],
            "signup_date": c.signup_date.date().isoformat(),
            "tenure_days": (datetime.utcnow() - c.signup_date).days,
            "tenure_tier": _tenure_tier(c.signup_date),
            "behavior_persona": c.behavior_persona,
            "total_orders": total_orders,
            "lifetime_spend": lifetime_spend,
            "loyalty_tier": _loyalty_tier(total_orders),
        }


# ---------------------------------------------------------------------------
# 2. Recent orders — with drift + recency signals
# ---------------------------------------------------------------------------
@tool
def get_customer_recent_orders(customer_id: int, n: int = 10) -> dict[str, Any]:
    """Get the customer's most recent orders (with merchant + cuisine info)
    plus signals: top cuisines, cuisine_drift (exploring vs repeating), and
    last_order_days_ago.

    Use this to personalize Smart Discovery ("you've been on a Thai run")
    or to flag exploration ("you tried Korean for the first time recently").

    Args:
        customer_id: The customer's numeric ID.
        n: How many recent orders to return (default 10, max 30).
    """
    n = max(1, min(n, 30))
    with get_session() as s:
        rows = s.execute(
            select(M.Order, M.Merchant)
            .join(M.Merchant, M.Order.merchant_id == M.Merchant.id)
            .where(M.Order.customer_id == customer_id)
            .order_by(M.Order.created_at.desc())
            .limit(n)
        ).all()

        if not rows:
            return {"customer_id": customer_id, "orders": [], "note": "no orders yet"}

        orders = [
            {
                "order_id": o.id,
                "merchant": m.name,
                "cuisine": m.cuisine,
                "total": o.total,
                "status": o.status,
                "created_at": o.created_at.isoformat(),
            }
            for o, m in rows
        ]

        # Top cuisines from this window
        cuisine_counter = Counter(m.cuisine for _, m in rows)
        top_cuisines = sorted(cuisine_counter.items(), key=lambda x: -x[1])

        # cuisine_drift: ratio of distinct cuisines in window vs total orders.
        # 1.0 = every order was a different cuisine (exploring)
        # 0.2 = mostly the same cuisine (repeating)
        distinct = len(cuisine_counter)
        drift_ratio = round(distinct / len(rows), 2) if rows else 0.0
        if drift_ratio >= 0.7:
            drift_label = "exploring"
        elif drift_ratio >= 0.4:
            drift_label = "mixed"
        else:
            drift_label = "repeating"

        # Recency
        latest_ts = rows[0][0].created_at
        last_order_days_ago = (datetime.utcnow() - latest_ts).days

        return {
            "customer_id": customer_id,
            "orders": orders,
            "total_orders_in_window": len(orders),
            "top_cuisines": [{"cuisine": c, "orders": n} for c, n in top_cuisines],
            "cuisine_drift_ratio": drift_ratio,
            "cuisine_drift": drift_label,
            "last_order_days_ago": last_order_days_ago,
        }


# ---------------------------------------------------------------------------
# 3. Typical pattern — the new anchor tool
# ---------------------------------------------------------------------------
@tool
def get_typical_pattern(customer_id: int) -> dict[str, Any]:
    """Compute the customer's typical ordering pattern — use this as the FIRST
    tool call to anchor every recommendation.

    Returns:
        - typical_hour_bucket: morning / lunch / afternoon / dinner / late-night
        - typical_hour_range: e.g. "19-21" (most-common 3-hour window)
        - weekday_share: fraction of orders Mon-Fri (0-1.0)
        - median_basket: typical price point
        - mean_basket: average price point
        - last_order_days_ago: recency signal
        - loyalty_tier: low / medium / high
        - favorite_cuisines: top 2 cuisines
        - favorite_merchant: most-ordered merchant (if any)
        - completion_rate: % of orders that completed (vs cancelled)

    Args:
        customer_id: The customer's numeric ID.
    """
    with get_session() as s:
        c: M.Customer | None = s.get(M.Customer, customer_id)
        if not c:
            return {"error": f"customer {customer_id} not found"}

        rows = s.execute(
            select(M.Order, M.Merchant)
            .join(M.Merchant, M.Order.merchant_id == M.Merchant.id)
            .where(M.Order.customer_id == customer_id)
            .order_by(M.Order.created_at.desc())
        ).all()

        if not rows:
            return {
                "customer_id": customer_id,
                "note": "no order history yet — agent should treat this as a new customer",
                "loyalty_tier": "low",
                "favorite_cuisines": [],
            }

        completed_orders = [(o, m) for o, m in rows if o.status == "completed"]
        completion_rate = round(len(completed_orders) / len(rows), 2)

        # Hours
        hours = [o.created_at.hour for o, _ in completed_orders]
        hour_counter = Counter(hours)
        most_common_hour = hour_counter.most_common(1)[0][0] if hours else 13
        typical_hour_bucket = _hour_bucket(most_common_hour)
        # 3-hour window centered on the mode
        h_lo = max(0, most_common_hour - 1)
        h_hi = min(23, most_common_hour + 1)
        typical_hour_range = f"{h_lo:02d}-{h_hi:02d}"

        # Weekdays
        weekdays = [o.created_at.weekday() < 5 for o, _ in completed_orders]
        weekday_share = round(sum(weekdays) / len(weekdays), 2) if weekdays else 0.0

        # Baskets
        totals = [o.total for o, _ in completed_orders]
        median_basket = round(median(totals), 2) if totals else 0.0
        mean_basket = round(sum(totals) / len(totals), 2) if totals else 0.0

        # Recency
        last_order_days_ago = (datetime.utcnow() - rows[0][0].created_at).days

        # Cuisines
        cuisine_counter = Counter(m.cuisine for _, m in completed_orders)
        favorite_cuisines = [c for c, _ in cuisine_counter.most_common(2)]

        # Favorite merchant
        merchant_counter = Counter(m.id for _, m in completed_orders)
        favorite_merchant: dict[str, Any] | None = None
        if merchant_counter:
            top_id, top_n = merchant_counter.most_common(1)[0]
            top_m = s.get(M.Merchant, top_id)
            if top_m and top_n >= 3:  # only call it a "favorite" if ordered 3+ times
                favorite_merchant = {
                    "merchant_id": top_m.id,
                    "name": top_m.name,
                    "cuisine": top_m.cuisine,
                    "rating": top_m.rating,
                    "orders": top_n,
                }

        return {
            "customer_id": customer_id,
            "typical_hour_bucket": typical_hour_bucket,
            "typical_hour_range": typical_hour_range,
            "weekday_share": weekday_share,
            "median_basket": median_basket,
            "mean_basket": mean_basket,
            "last_order_days_ago": last_order_days_ago,
            "completion_rate": completion_rate,
            "loyalty_tier": _loyalty_tier(len(completed_orders)),
            "favorite_cuisines": favorite_cuisines,
            "favorite_merchant": favorite_merchant,
            "behavior_persona": c.behavior_persona,
            "dietary_prefs": c.dietary_prefs or [],
            "city": c.city.name,
        }


# ---------------------------------------------------------------------------
# 4. Search merchants
# ---------------------------------------------------------------------------
@tool
def search_merchants(
    city_name: str,
    cuisine: str | None = None,
    dietary_filter: str | None = None,
    max_prep_min: int | None = None,
    limit: int = 6,
) -> dict[str, Any]:
    """Find merchants in a city matching cuisine + dietary + prep-time filters.

    Returns the top-rated matches with a couple of sample menu items each. Use
    this for the Smart Discovery pillar.

    Args:
        city_name: City name (e.g. "Singapore").
        cuisine: Optional cuisine filter (e.g. "Thai", "Indian", "Vegetarian").
        dietary_filter: Optional dietary tag — only returns merchants that have
            at least one menu item with this tag. One of "vegetarian", "vegan",
            "halal", "gluten-free".
        max_prep_min: Optional cap on average prep time.
        limit: Max merchants to return (default 6).
    """
    limit = max(1, min(limit, 20))
    with get_session() as s:
        city = s.scalar(select(M.City).where(M.City.name == city_name))
        if not city:
            return {"error": f"city {city_name!r} not found"}

        filters = [M.Merchant.city_id == city.id]
        if cuisine:
            filters.append(M.Merchant.cuisine == cuisine)
        if max_prep_min is not None:
            filters.append(M.Merchant.avg_prep_min <= max_prep_min)

        merchants = s.execute(
            select(M.Merchant)
            .where(and_(*filters))
            .order_by(M.Merchant.rating.desc())
            .limit(limit * 3 if dietary_filter else limit)
        ).scalars().all()

        results: list[dict[str, Any]] = []
        for m in merchants:
            items = m.menu_items
            if dietary_filter:
                matching_items = [i for i in items if dietary_filter in (i.tags or [])]
                if not matching_items:
                    continue
                matching_items.sort(key=lambda i: -i.popularity)
                highlights = matching_items[:2]
            else:
                highlights = sorted(items, key=lambda i: -i.popularity)[:2]

            results.append({
                "merchant_id": m.id,
                "name": m.name,
                "cuisine": m.cuisine,
                "rating": m.rating,
                "zone": m.zone,
                "avg_prep_min": m.avg_prep_min,
                "highlights": [
                    {"name": h.name, "price": h.price, "tags": h.tags or []}
                    for h in highlights
                ],
            })
            if len(results) >= limit:
                break

        return {
            "city": city_name,
            "filters": {"cuisine": cuisine, "dietary": dietary_filter, "max_prep_min": max_prep_min},
            "matches": results,
            "count": len(results),
        }


# ---------------------------------------------------------------------------
# 5. Get merchant menu
# ---------------------------------------------------------------------------
@tool
def get_merchant_menu(merchant_id: int, dietary_filter: str | None = None) -> dict[str, Any]:
    """Return the full menu (or a dietary-filtered subset) for one merchant.

    Args:
        merchant_id: The merchant's numeric ID.
        dietary_filter: Optional, one of "vegetarian", "vegan", "halal", "gluten-free".
    """
    with get_session() as s:
        m: M.Merchant | None = s.get(M.Merchant, merchant_id)
        if not m:
            return {"error": f"merchant {merchant_id} not found"}

        items = list(m.menu_items)
        if dietary_filter:
            items = [i for i in items if dietary_filter in (i.tags or [])]
        items.sort(key=lambda i: -i.popularity)

        return {
            "merchant_id": m.id,
            "merchant_name": m.name,
            "cuisine": m.cuisine,
            "rating": m.rating,
            "filter": dietary_filter,
            "items": [
                {
                    "name": i.name,
                    "price": i.price,
                    "tags": i.tags or [],
                    "popularity": i.popularity,
                }
                for i in items
            ],
            "count": len(items),
        }


# ---------------------------------------------------------------------------
# 6. Safe late-night drivers
# ---------------------------------------------------------------------------
@tool
def find_safe_late_night_drivers(
    city_name: str,
    vehicle_type: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Find the highest-trust drivers in a city for late-night deliveries/rides.

    Filters: rating ≥ 4.7, cancel_rate ≤ 0.05. Sorted by rating desc, cancel
    rate asc. Use this for the Safe Late-Night Matching pillar.

    Args:
        city_name: City name.
        vehicle_type: Optional, "bike" or "car".
        limit: Max drivers to return (default 5).
    """
    limit = max(1, min(limit, 20))
    with get_session() as s:
        city = s.scalar(select(M.City).where(M.City.name == city_name))
        if not city:
            return {"error": f"city {city_name!r} not found"}

        filters = [
            M.Driver.city_id == city.id,
            M.Driver.is_active.is_(True),
            M.Driver.rating >= 4.7,
            M.Driver.cancel_rate <= 0.05,
        ]
        if vehicle_type:
            filters.append(M.Driver.vehicle_type == vehicle_type)

        drivers = s.execute(
            select(M.Driver)
            .where(and_(*filters))
            .order_by(M.Driver.rating.desc(), M.Driver.cancel_rate.asc())
            .limit(limit)
        ).scalars().all()

        return {
            "city": city_name,
            "criteria": {
                "min_rating": 4.7,
                "max_cancel_rate": 0.05,
                "vehicle_type": vehicle_type,
            },
            "drivers": [
                {
                    "driver_id": d.id,
                    "name": d.name,
                    "vehicle_type": d.vehicle_type,
                    "rating": d.rating,
                    "cancel_rate": d.cancel_rate,
                    "tenure_days": (datetime.utcnow() - d.joined_date).days,
                }
                for d in drivers
            ],
            "count": len(drivers),
        }


CUSTOMER_TOOLS = [
    get_customer_profile,
    get_customer_recent_orders,
    get_typical_pattern,
    search_merchants,
    get_merchant_menu,
    find_safe_late_night_drivers,
]
