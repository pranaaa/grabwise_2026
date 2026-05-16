"""Tools the Merchant Growth Agent calls.

Pillars from the deck:
  • AI Pricing & Discount Suggestions — recommend discounts, bundles, promotions to lift conversions.
  • Demand Forecasting & Trend Insights — predict order volumes by time and location.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select, func, and_
from langchain_core.tools import tool

from backend.db.database import get_session
from backend.db import models as M


_DOW = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}


# --------------------------- 1. Merchant profile -----------------------------
@tool
def get_merchant_profile(merchant_id: int) -> dict[str, Any]:
    """Look up a merchant's profile (name, city, zone, cuisine, rating, prep time, menu size).

    Args:
        merchant_id: The merchant's numeric ID.
    """
    with get_session() as s:
        m: M.Merchant | None = s.get(M.Merchant, merchant_id)
        if not m:
            return {"error": f"merchant {merchant_id} not found"}
        return {
            "merchant_id": m.id,
            "name": m.name,
            "city": m.city.name,
            "zone": m.zone,
            "cuisine": m.cuisine,
            "rating": m.rating,
            "avg_prep_min": m.avg_prep_min,
            "menu_items_count": len(m.menu_items),
        }


# --------------------------- 2. Order rollup ---------------------------------
@tool
def get_merchant_order_rollup(merchant_id: int, days: int = 30) -> dict[str, Any]:
    """Orders, revenue, AOV, completion rate, and weekday-vs-weekend split.

    Args:
        merchant_id: Merchant's numeric ID.
        days: Lookback window in days (default 30, max 90).
    """
    days = max(7, min(days, 90))
    cutoff = datetime.utcnow() - timedelta(days=days)
    with get_session() as s:
        rows = s.execute(
            select(M.Order).where(
                M.Order.merchant_id == merchant_id,
                M.Order.created_at >= cutoff,
            )
        ).scalars().all()

    if not rows:
        return {
            "merchant_id": merchant_id,
            "lookback_days": days,
            "note": "no orders in this window",
        }

    completed = [o for o in rows if o.status == "completed"]
    cancelled = [o for o in rows if o.status == "cancelled"]
    revenue = sum(o.total for o in completed)
    aov = revenue / len(completed) if completed else 0
    completion_rate = len(completed) / len(rows) if rows else 0

    # Weekend vs weekday
    weekend_completed = [o for o in completed if o.created_at.weekday() >= 5]
    weekday_completed = [o for o in completed if o.created_at.weekday() < 5]
    weekend_orders_per_day = len(weekend_completed) / max(1, days * 2 / 7)
    weekday_orders_per_day = len(weekday_completed) / max(1, days * 5 / 7)

    return {
        "merchant_id": merchant_id,
        "lookback_days": days,
        "total_orders": len(rows),
        "completed_orders": len(completed),
        "cancelled_orders": len(cancelled),
        "revenue": round(revenue, 2),
        "avg_order_value": round(aov, 2),
        "completion_rate_pct": round(completion_rate * 100, 1),
        "weekend_orders_per_day": round(weekend_orders_per_day, 2),
        "weekday_orders_per_day": round(weekday_orders_per_day, 2),
        "weekend_vs_weekday_lift_pct": round(
            (weekend_orders_per_day / weekday_orders_per_day - 1) * 100, 1
        ) if weekday_orders_per_day else 0.0,
    }


# --------------------------- 3. Top items ------------------------------------
@tool
def get_top_items(merchant_id: int, limit: int = 5) -> dict[str, Any]:
    """Return the merchant's most popular menu items (by stored popularity score).

    Use this to understand what's driving sales — input for pricing/bundle decisions.

    Args:
        merchant_id: Merchant's numeric ID.
        limit: Max items to return (default 5).
    """
    limit = max(1, min(limit, 20))
    with get_session() as s:
        m: M.Merchant | None = s.get(M.Merchant, merchant_id)
        if not m:
            return {"error": f"merchant {merchant_id} not found"}
        items = sorted(m.menu_items, key=lambda i: -i.popularity)[:limit]
        return {
            "merchant_id": merchant_id,
            "merchant_name": m.name,
            "top_items": [
                {
                    "name": i.name,
                    "price": i.price,
                    "popularity": i.popularity,
                    "tags": i.tags or [],
                }
                for i in items
            ],
        }


# --------------------------- 4. Demand forecast ------------------------------
@tool
def forecast_merchant_demand(
    merchant_id: int,
    day_of_week: str | None = None,
    hour: int | None = None,
) -> dict[str, Any]:
    """Predict expected order volume for a given day-of-week and hour window.

    Uses 8-week historical orders for this merchant, hour smoothed by ±1.
    The deck's "Demand Forecasting & Trend Insights" pillar.

    Args:
        merchant_id: Merchant's numeric ID.
        day_of_week: Optional "Mon"|...|"Sun".
        hour: Optional 0-23 (window is ±1 hour).
    """
    target_dow = _DOW.get(day_of_week) if day_of_week else None
    cutoff = datetime.utcnow() - timedelta(days=56)

    with get_session() as s:
        rows = s.execute(
            select(M.Order.created_at).where(
                M.Order.merchant_id == merchant_id,
                M.Order.created_at >= cutoff,
                M.Order.status == "completed",
            )
        ).all()

    matching = []
    for (ts,) in rows:
        if target_dow is not None and ts.weekday() != target_dow:
            continue
        if hour is not None and not (hour - 1 <= ts.hour <= hour + 1):
            continue
        matching.append(ts)

    if not matching:
        return {
            "merchant_id": merchant_id,
            "filters": {"day_of_week": day_of_week, "hour": hour},
            "expected_orders_per_window": 0.0,
            "note": "not enough history for this filter",
        }

    distinct_dates = {ts.date() for ts in matching}
    expected_per_window = len(matching) / max(1, len(distinct_dates))

    # Sanity reference: same lookback overall avg orders per relevant DOW
    return {
        "merchant_id": merchant_id,
        "filters": {"day_of_week": day_of_week, "hour": hour},
        "lookback_days": 56,
        "matching_orders": len(matching),
        "days_observed": len(distinct_dates),
        "expected_orders_per_window": round(expected_per_window, 2),
    }


# --------------------------- 5. Competitor signals ---------------------------
@tool
def get_competitor_signals(merchant_id: int, limit: int = 5) -> dict[str, Any]:
    """Find same-cuisine merchants in the same city (peers/competitors) and surface
    their ratings + a top item from each.

    Use this to benchmark pricing and surface menu ideas.

    Args:
        merchant_id: Merchant's numeric ID.
        limit: Max competitors to return (default 5).
    """
    limit = max(1, min(limit, 20))
    with get_session() as s:
        m: M.Merchant | None = s.get(M.Merchant, merchant_id)
        if not m:
            return {"error": f"merchant {merchant_id} not found"}

        peers = s.execute(
            select(M.Merchant)
            .where(
                M.Merchant.city_id == m.city_id,
                M.Merchant.cuisine == m.cuisine,
                M.Merchant.id != m.id,
            )
            .order_by(M.Merchant.rating.desc())
            .limit(limit)
        ).scalars().all()

        results = []
        for p in peers:
            top = sorted(p.menu_items, key=lambda i: -i.popularity)[:1]
            results.append({
                "merchant_id": p.id,
                "name": p.name,
                "rating": p.rating,
                "zone": p.zone,
                "top_item": {
                    "name": top[0].name,
                    "price": top[0].price,
                } if top else None,
            })

        # Compute peer-rating benchmark
        peer_avg_rating = round(sum(p.rating for p in peers) / len(peers), 2) if peers else None

        return {
            "merchant_id": merchant_id,
            "your_rating": m.rating,
            "your_cuisine": m.cuisine,
            "peer_count": len(peers),
            "peer_avg_rating": peer_avg_rating,
            "rating_gap_vs_peers": round(m.rating - (peer_avg_rating or m.rating), 2) if peer_avg_rating else None,
            "competitors": results,
        }


# --------------------------- 6. Pricing actions ------------------------------
@tool
def suggest_pricing_actions(merchant_id: int) -> dict[str, Any]:
    """Analyze the merchant's menu for pricing/discount/bundle opportunities.

    Heuristics:
      - High-popularity + below-cuisine-median price → candidate for a small price uplift.
      - Low-popularity + above-cuisine-median price → candidate for a discount or removal.
      - Top item + a low-popularity complement → bundle opportunity.

    Returns structured signals; the agent narrates them as concrete suggestions.

    Args:
        merchant_id: Merchant's numeric ID.
    """
    with get_session() as s:
        m: M.Merchant | None = s.get(M.Merchant, merchant_id)
        if not m:
            return {"error": f"merchant {merchant_id} not found"}
        items = list(m.menu_items)
        if not items:
            return {"merchant_id": merchant_id, "note": "no menu items to analyze"}

        # Cuisine-wide median price (across all merchants of same cuisine in same city)
        peer_items = s.execute(
            select(M.MenuItem.price).join(M.Merchant).where(
                M.Merchant.city_id == m.city_id,
                M.Merchant.cuisine == m.cuisine,
            )
        ).all()
        peer_prices = sorted([p[0] for p in peer_items])
        cuisine_median = peer_prices[len(peer_prices) // 2] if peer_prices else None

        sorted_by_pop = sorted(items, key=lambda i: -i.popularity)
        median_pop = sorted(i.popularity for i in items)[len(items) // 2]

        uplift_candidates: list[dict[str, Any]] = []
        discount_candidates: list[dict[str, Any]] = []
        for i in items:
            if cuisine_median is None:
                break
            if i.popularity >= median_pop and i.price < cuisine_median * 0.9:
                uplift_candidates.append({
                    "item": i.name,
                    "price": i.price,
                    "popularity": i.popularity,
                    "cuisine_median": round(cuisine_median, 2),
                    "suggested_action": f"raise price toward ${round(cuisine_median, 2)} (high demand, below market)",
                })
            elif i.popularity <= median_pop and i.price > cuisine_median * 1.1:
                discount_candidates.append({
                    "item": i.name,
                    "price": i.price,
                    "popularity": i.popularity,
                    "cuisine_median": round(cuisine_median, 2),
                    "suggested_action": f"discount 10-15% to test demand (low pop, above market)",
                })

        # Bundle: top item + a low-popularity item
        bundle_suggestion = None
        if len(sorted_by_pop) >= 2:
            top = sorted_by_pop[0]
            laggard = sorted_by_pop[-1]
            if laggard.popularity < top.popularity * 0.5:
                bundle_suggestion = {
                    "anchor": top.name,
                    "anchor_price": top.price,
                    "add_on": laggard.name,
                    "add_on_price": laggard.price,
                    "bundle_target_price": round((top.price + laggard.price) * 0.9, 2),
                    "rationale": f"pair top seller ({top.name}) with low-mover ({laggard.name}) at ~10% off",
                }

        return {
            "merchant_id": merchant_id,
            "cuisine_median_price": round(cuisine_median, 2) if cuisine_median else None,
            "uplift_candidates": uplift_candidates[:3],
            "discount_candidates": discount_candidates[:3],
            "bundle_suggestion": bundle_suggestion,
        }


MERCHANT_TOOLS = [
    get_merchant_profile,
    get_merchant_order_rollup,
    get_top_items,
    forecast_merchant_demand,
    get_competitor_signals,
    suggest_pricing_actions,
]
