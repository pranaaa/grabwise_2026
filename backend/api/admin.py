"""Admin-only ecosystem dashboard endpoints."""
from __future__ import annotations
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func

from backend.db.database import get_session
from backend.db import models as M
from backend.api.auth import require_admin, CurrentUser
from backend.tools._risk_math import (
    compute_customer_anomaly_score,
    compute_order_risk_score,
    compute_driver_trust_score,
)

router = APIRouter(prefix="/api/admin", tags=["admin"])

CITY_NAMES = ["Singapore", "Jakarta", "Bangkok", "Manila", "Kuala Lumpur"]


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


def _city_id_for_name(s, name: str) -> int | None:
    if name == "All":
        return None
    row = s.execute(select(M.City).where(M.City.name == name)).scalar_one_or_none()
    return row.id if row else None


@router.get("/cities")
def cities(_admin: CurrentUser = Depends(require_admin)) -> dict[str, list[str]]:
    return {"cities": ["All"] + CITY_NAMES}


# ---------------------------------------------------------------------------
# LLM observability — per-model performance + cost rollup
# ---------------------------------------------------------------------------
@router.get("/llm-stats")
def llm_stats(
    _admin: CurrentUser = Depends(require_admin),
    hours: int = Query(default=24 * 7, ge=1, le=24 * 90),
) -> dict[str, Any]:
    """Per-model usage, hallucination, and cost rollup for the last `hours` hours.

    Used by the admin "LLM Performance & Cost" widget. Numbers come from the
    `llm_call_logs` table written by backend.observability.tracker.
    """
    from backend.llm.registry import BEDROCK_MODELS, find as find_model
    cutoff = datetime.utcnow() - timedelta(hours=hours)

    with get_session() as s:
        rows = s.execute(
            select(M.LLMCallLog).where(M.LLMCallLog.invoked_at >= cutoff)
        ).scalars().all()

    by_model: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "calls": 0,
        "hallucinations": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "total_latency_ms": 0,
        "latency_samples": 0,
        "reason_counts": defaultdict(int),
        "agents": defaultdict(int),
    })
    for r in rows:
        b = by_model[r.model_id]
        b["calls"] += 1
        if r.hallucinated:
            b["hallucinations"] += 1
        if r.input_tokens:
            b["input_tokens"] += int(r.input_tokens)
        if r.output_tokens:
            b["output_tokens"] += int(r.output_tokens)
        if r.estimated_cost_usd:
            b["cost_usd"] += float(r.estimated_cost_usd)
        if r.duration_ms:
            b["total_latency_ms"] += int(r.duration_ms)
            b["latency_samples"] += 1
        for code in (r.reasons_json or []):
            b["reason_counts"][code.split(":", 1)[0]] += 1
        if r.agent:
            b["agents"][r.agent] += 1

    # Verdict heuristic
    def verdict(stats: dict[str, Any], hall_rate: float, cost: float) -> dict[str, str]:
        if stats["calls"] < 5:
            return {"label": "Insufficient data", "tone": "neutral"}
        if hall_rate >= 0.30:
            return {"label": "Unreliable", "tone": "danger"}
        if hall_rate >= 0.10:
            return {"label": "Flaky", "tone": "warn"}
        if cost > 0 and (cost / max(stats["calls"], 1)) > 0.02:
            return {"label": "Pricey but reliable", "tone": "warn"}
        return {"label": "Reliable", "tone": "good"}

    models: list[dict[str, Any]] = []
    for model_id, stats in by_model.items():
        hall_rate = stats["hallucinations"] / stats["calls"] if stats["calls"] > 0 else 0.0
        avg_lat = (stats["total_latency_ms"] / stats["latency_samples"]) if stats["latency_samples"] else None
        reg = find_model(model_id)
        v = verdict(stats, hall_rate, stats["cost_usd"])
        models.append({
            "model_id": model_id,
            "model_name": reg["name"] if reg else model_id,
            "category": reg["category"] if reg else "Other",
            "size": reg["size"] if reg else None,
            "calls": stats["calls"],
            "hallucinations": stats["hallucinations"],
            "hallucination_rate": round(hall_rate, 3),
            "input_tokens": stats["input_tokens"],
            "output_tokens": stats["output_tokens"],
            "estimated_cost_usd": round(stats["cost_usd"], 4),
            "cost_per_call_usd": round(stats["cost_usd"] / stats["calls"], 5) if stats["calls"] else None,
            "avg_latency_ms": round(avg_lat) if avg_lat is not None else None,
            "top_reasons": sorted(stats["reason_counts"].items(), key=lambda x: -x[1])[:3],
            "agents": dict(stats["agents"]),
            "verdict": v,
        })
    models.sort(key=lambda m: -m["calls"])

    # Aggregate totals
    totals = {
        "calls": sum(m["calls"] for m in models),
        "hallucinations": sum(m["hallucinations"] for m in models),
        "input_tokens": sum(m["input_tokens"] for m in models),
        "output_tokens": sum(m["output_tokens"] for m in models),
        "estimated_cost_usd": round(sum(m["estimated_cost_usd"] for m in models), 4),
    }
    totals["hallucination_rate"] = (
        round(totals["hallucinations"] / totals["calls"], 3) if totals["calls"] else 0.0
    )

    return {
        "window_hours": hours,
        "total": totals,
        "models": models,
    }


@router.get("/kpis")
def kpis(
    city: str = Query(default="All"),
    _admin: CurrentUser = Depends(require_admin),
) -> dict[str, Any]:
    now = datetime.utcnow()
    window_days = 7
    cutoff_now = now - timedelta(days=window_days)
    cutoff_prev_start = now - timedelta(days=window_days * 2)
    cutoff_prev_end = cutoff_now
    cutoff_active = now - timedelta(days=14)
    cutoff_spark = now - timedelta(days=14)

    with get_session() as s:
        city_id = _city_id_for_name(s, city)

        base_filter = []
        if city_id is not None:
            base_filter.append(M.Order.city_id == city_id)

        # Pull last 14 days of orders once for both window + sparkline.
        rows = s.execute(
            select(M.Order).where(M.Order.created_at >= cutoff_spark, *base_filter)
        ).scalars().all()
        # And the prior 7d window.
        prev_rows = s.execute(
            select(M.Order).where(
                M.Order.created_at >= cutoff_prev_start,
                M.Order.created_at < cutoff_prev_end,
                *base_filter,
            )
        ).scalars().all()

        completed_now = [o for o in rows if o.status == "completed" and o.created_at >= cutoff_now]
        all_now = [o for o in rows if o.created_at >= cutoff_now]
        completed_prev = [o for o in prev_rows if o.status == "completed"]
        all_prev = list(prev_rows)

        # Active drivers/merchants (14d).
        active_drivers = {o.driver_id for o in rows if o.status == "completed" and o.driver_id is not None}
        active_merchants = {o.merchant_id for o in rows if o.status == "completed"}
        # Prior 14d (28d→14d ago) for active drivers/merchants delta.
        prev_14d = s.execute(
            select(M.Order).where(
                M.Order.created_at >= now - timedelta(days=28),
                M.Order.created_at < cutoff_active,
                *base_filter,
            )
        ).scalars().all()
        prev_active_drivers = {o.driver_id for o in prev_14d if o.status == "completed" and o.driver_id is not None}
        prev_active_merchants = {o.merchant_id for o in prev_14d if o.status == "completed"}

        # GMV
        gmv_now = sum(o.total for o in completed_now)
        gmv_prev = sum(o.total for o in completed_prev)
        # AOV
        aov_now = (gmv_now / len(completed_now)) if completed_now else 0.0
        aov_prev = (gmv_prev / len(completed_prev)) if completed_prev else 0.0
        # Cancel rate
        cancel_now = (
            sum(1 for o in all_now if o.status == "cancelled") / len(all_now)
        ) if all_now else 0.0
        cancel_prev = (
            sum(1 for o in all_prev if o.status == "cancelled") / len(all_prev)
        ) if all_prev else 0.0
        # Fraud-flagged %
        fraud_now = (
            sum(1 for o in all_now if (o.risk_score or 0) >= 60) / len(all_now)
        ) if all_now else 0.0
        fraud_prev = (
            sum(1 for o in all_prev if (o.risk_score or 0) >= 60) / len(all_prev)
        ) if all_prev else 0.0

        def pct_delta(curr: float, prev: float) -> float:
            if prev == 0:
                return 0.0 if curr == 0 else 100.0
            return round(((curr - prev) / prev) * 100, 1)

        def signed_delta(curr: float, prev: float) -> float:
            return round(curr - prev, 1)

        # Sparklines: per-day for the last 14 days, oldest -> newest.
        days = [(now - timedelta(days=i)).date() for i in range(13, -1, -1)]
        by_day_completed: dict[Any, list[M.Order]] = defaultdict(list)
        by_day_all: dict[Any, list[M.Order]] = defaultdict(list)
        for o in rows:
            d = o.created_at.date()
            by_day_all[d].append(o)
            if o.status == "completed":
                by_day_completed[d].append(o)
        spark_gmv = [round(sum(o.total for o in by_day_completed.get(d, [])), 2) for d in days]
        spark_aov = [
            round((sum(o.total for o in by_day_completed.get(d, [])) / len(by_day_completed[d])), 2)
            if by_day_completed.get(d) else 0.0
            for d in days
        ]
        spark_cancel = [
            round(sum(1 for o in by_day_all.get(d, []) if o.status == "cancelled") / len(by_day_all[d]), 4)
            if by_day_all.get(d) else 0.0
            for d in days
        ]
        spark_fraud = [
            round(sum(1 for o in by_day_all.get(d, []) if (o.risk_score or 0) >= 60) / len(by_day_all[d]), 4)
            if by_day_all.get(d) else 0.0
            for d in days
        ]
        spark_drivers: list[float] = []
        spark_merchants: list[float] = []
        # 7-day rolling unique active drivers/merchants ending each day
        for d in days:
            window_start = d - timedelta(days=6)
            ds = set()
            ms = set()
            for o in rows:
                od = o.created_at.date()
                if window_start <= od <= d and o.status == "completed":
                    if o.driver_id is not None:
                        ds.add(o.driver_id)
                    ms.add(o.merchant_id)
            spark_drivers.append(len(ds))
            spark_merchants.append(len(ms))

        kpis_list = [
            {
                "id": "gmv",
                "label": "GMV (7d)",
                "value": _fmt_currency(gmv_now),
                "value_raw": round(gmv_now, 2),
                "delta_pct": pct_delta(gmv_now, gmv_prev),
                "fmt": "currency",
                "direction": "higher_is_better",
                "spark": spark_gmv,
            },
            {
                "id": "active_drivers",
                "label": "Active Drivers",
                "value": _fmt_int(len(active_drivers)),
                "value_raw": len(active_drivers),
                "delta_pct": pct_delta(len(active_drivers), len(prev_active_drivers)),
                "fmt": "int",
                "direction": "higher_is_better",
                "spark": spark_drivers,
            },
            {
                "id": "active_merchants",
                "label": "Active Merchants",
                "value": _fmt_int(len(active_merchants)),
                "value_raw": len(active_merchants),
                "delta_pct": pct_delta(len(active_merchants), len(prev_active_merchants)),
                "fmt": "int",
                "direction": "higher_is_better",
                "spark": spark_merchants,
            },
            {
                "id": "aov",
                "label": "Avg Order Value",
                "value": _fmt_currency(aov_now),
                "value_raw": round(aov_now, 2),
                "delta_pct": pct_delta(aov_now, aov_prev),
                "fmt": "currency",
                "direction": "higher_is_better",
                "spark": spark_aov,
            },
            {
                "id": "cancel_rate",
                "label": "Cancel Rate",
                "value": _fmt_percent(cancel_now),
                "value_raw": round(cancel_now, 4),
                "delta_pct": signed_delta(cancel_now * 100, cancel_prev * 100),
                "fmt": "percent",
                "direction": "lower_is_better",
                "spark": spark_cancel,
            },
            {
                "id": "fraud_pct",
                "label": "Fraud-Flagged %",
                "value": _fmt_percent(fraud_now),
                "value_raw": round(fraud_now, 4),
                "delta_pct": signed_delta(fraud_now * 100, fraud_prev * 100),
                "fmt": "percent",
                "direction": "lower_is_better",
                "spark": spark_fraud,
            },
        ]
        return {
            "window_days": window_days,
            "active_city": city,
            "kpis": kpis_list,
        }


@router.get("/heatmap")
def heatmap(
    city: str = Query(default="All"),
    days: int = Query(default=30),
    _admin: CurrentUser = Depends(require_admin),
) -> dict[str, Any]:
    now = datetime.utcnow()
    cutoff = now - timedelta(days=days)

    with get_session() as s:
        city_id = _city_id_for_name(s, city)
        if city_id is None:
            zones: list[str] = []
            for c in s.execute(select(M.City).order_by(M.City.id)).scalars().all():
                zones.extend(c.zones)
        else:
            row = s.get(M.City, city_id)
            zones = list(row.zones) if row else []

        filters = [M.Order.created_at >= cutoff, M.Order.status == "completed"]
        if city_id is not None:
            filters.append(M.Order.city_id == city_id)
        rows = s.execute(select(M.Order).where(*filters)).scalars().all()

        zone_idx = {z: i for i, z in enumerate(zones)}
        matrix = [[0] * 24 for _ in zones]
        for o in rows:
            i = zone_idx.get(o.pickup_zone)
            if i is None:
                continue
            matrix[i][o.created_at.hour] += 1
        flat = [v for row in matrix for v in row]
        max_val = max(flat) if flat else 0

        return {
            "zones": zones,
            "hours": list(range(24)),
            "matrix": matrix,
            "max": max_val,
            "active_city": city,
            "days": days,
        }


@router.get("/driver-earnings")
def driver_earnings(
    city: str = Query(default="All"),
    days: int = Query(default=30),
    bins: int = Query(default=8),
    _admin: CurrentUser = Depends(require_admin),
) -> dict[str, Any]:
    now = datetime.utcnow()
    cutoff = now - timedelta(days=days)

    with get_session() as s:
        city_id = _city_id_for_name(s, city)
        filters = [
            M.Order.created_at >= cutoff,
            M.Order.status == "completed",
            M.Order.driver_id.is_not(None),
        ]
        if city_id is not None:
            filters.append(M.Order.city_id == city_id)

        rows = s.execute(select(M.Order).where(*filters)).scalars().all()
        per_driver: dict[int, float] = defaultdict(float)
        for o in rows:
            per_driver[o.driver_id] += o.driver_earning or 0.0

        earnings = sorted(per_driver.values())
        n = len(earnings)
        if n == 0:
            return {
                "bins": [],
                "total_drivers": 0,
                "p50": 0.0,
                "p90": 0.0,
                "low_cohort_threshold": 0.0,
                "low_cohort_count": 0,
                "active_city": city,
            }

        def pct(arr: list[float], q: float) -> float:
            if not arr:
                return 0.0
            idx = max(0, min(len(arr) - 1, int(round((len(arr) - 1) * q))))
            return round(arr[idx], 2)

        p50 = pct(earnings, 0.50)
        p90 = pct(earnings, 0.90)
        low_threshold = pct(earnings, 0.20)
        low_count = sum(1 for v in earnings if v <= low_threshold)

        lo = 0.0
        hi = max(earnings) if earnings else 0.0
        if hi <= lo:
            hi = lo + 1.0
        width = (hi - lo) / max(bins, 1)
        bin_counts = [0] * bins
        for v in earnings:
            i = int((v - lo) / width)
            if i >= bins:
                i = bins - 1
            if i < 0:
                i = 0
            bin_counts[i] += 1
        bins_out = [
            {"lo": round(lo + i * width, 2), "hi": round(lo + (i + 1) * width, 2), "count": c}
            for i, c in enumerate(bin_counts)
        ]

        return {
            "bins": bins_out,
            "total_drivers": n,
            "p50": p50,
            "p90": p90,
            "low_cohort_threshold": round(low_threshold, 2),
            "low_cohort_count": low_count,
            "active_city": city,
        }


# ============================================================================
# Layer 3 — Fraud & Risk Console
# ============================================================================

DECISION_VALUES = ("approve", "review", "block")


@router.get("/risk-feed")
def risk_feed(
    city: str = Query(default="All"),
    decision: str = Query(default="all"),
    limit: int = Query(default=50, ge=1, le=200),
    _admin: CurrentUser = Depends(require_admin),
) -> dict[str, Any]:
    decision = decision.lower()
    if decision not in ("all", *DECISION_VALUES):
        decision = "all"

    with get_session() as s:
        city_id = _city_id_for_name(s, city)
        base_filters = []
        if city_id is not None:
            base_filters.append(M.Order.city_id == city_id)

        # Chip counts always reflect city filter, ignoring decision filter.
        count_rows = s.execute(
            select(M.Order.risk_decision, func.count(M.Order.id)).where(*base_filters).group_by(M.Order.risk_decision)
        ).all()
        decision_counts = {"approve": 0, "review": 0, "block": 0, "all": 0}
        for d, c in count_rows:
            if d in DECISION_VALUES:
                decision_counts[d] = int(c)
            decision_counts["all"] += int(c)

        feed_filters = list(base_filters)
        if decision != "all":
            feed_filters.append(M.Order.risk_decision == decision)

        rows = s.execute(
            select(M.Order)
            .where(*feed_filters)
            .order_by(M.Order.created_at.desc())
            .limit(limit)
        ).scalars().all()

        # Bulk-fetch related entities to avoid N+1 lazy loads.
        cust_ids = {o.customer_id for o in rows}
        merch_ids = {o.merchant_id for o in rows}
        drv_ids = {o.driver_id for o in rows if o.driver_id is not None}
        city_ids = {o.city_id for o in rows}

        customers = {
            c.id: c for c in s.execute(
                select(M.Customer).where(M.Customer.id.in_(cust_ids))
            ).scalars().all()
        } if cust_ids else {}
        merchants = {
            m.id: m for m in s.execute(
                select(M.Merchant).where(M.Merchant.id.in_(merch_ids))
            ).scalars().all()
        } if merch_ids else {}
        drivers = {
            d.id: d for d in s.execute(
                select(M.Driver).where(M.Driver.id.in_(drv_ids))
            ).scalars().all()
        } if drv_ids else {}
        cities = {
            c.id: c for c in s.execute(
                select(M.City).where(M.City.id.in_(city_ids))
            ).scalars().all()
        } if city_ids else {}

        orders_out: list[dict[str, Any]] = []
        for o in rows:
            hour = o.created_at.hour
            late_night = hour >= 22 or hour < 5
            cust = customers.get(o.customer_id)
            merch = merchants.get(o.merchant_id)
            drv = drivers.get(o.driver_id) if o.driver_id is not None else None
            cty = cities.get(o.city_id)
            orders_out.append({
                "id": o.id,
                "created_at": o.created_at.isoformat(),
                "city": cty.name if cty else "",
                "customer_id": o.customer_id,
                "customer_name": cust.name if cust else f"Customer #{o.customer_id}",
                "merchant_id": o.merchant_id,
                "merchant_name": merch.name if merch else f"Merchant #{o.merchant_id}",
                "driver_id": o.driver_id,
                "driver_name": drv.name if drv else None,
                "total": round(o.total, 2),
                "hour": hour,
                "late_night": late_night,
                "pickup_zone": o.pickup_zone,
                "dropoff_zone": o.dropoff_zone,
                "status": o.status,
                "risk_score": round(o.risk_score, 1) if o.risk_score is not None else None,
                "risk_decision": o.risk_decision,
            })

        return {
            "active_city": city,
            "active_decision": decision,
            "decision_counts": decision_counts,
            "orders": orders_out,
        }


def _customer_anomaly_payload(s, c: M.Customer, now: datetime) -> dict[str, Any]:
    tenure_days = (now - c.signup_date).days
    recent_cutoff = now - timedelta(days=7)

    recent_orders = s.execute(
        select(M.Order).where(
            M.Order.customer_id == c.id,
            M.Order.created_at >= recent_cutoff,
        )
    ).scalars().all()
    recent_n = len(recent_orders)
    recent_cancels = sum(1 for o in recent_orders if o.status == "cancelled")

    all_orders = s.execute(
        select(M.Order).where(M.Order.customer_id == c.id)
    ).scalars().all()
    avg_total = (sum(o.total for o in all_orders) / len(all_orders)) if all_orders else 0.0

    anomaly_score, flags = compute_customer_anomaly_score(
        tenure_days=tenure_days,
        recent_orders_7d=recent_n,
        recent_cancels_7d=recent_cancels,
    )
    return {
        "id": c.id,
        "name": c.name,
        "city": c.city.name if c.city else "",
        "tenure_days": tenure_days,
        "signup_date": c.signup_date.isoformat(),
        "total_orders": len(all_orders),
        "avg_order_value": round(avg_total, 2),
        "recent_orders_7d": recent_n,
        "recent_cancels_7d": recent_cancels,
        "anomaly_score": anomaly_score,
        "anomaly_flags": flags or ["no anomalies detected"],
    }


def _driver_trust_payload(s, d: M.Driver, now: datetime) -> dict[str, Any]:
    tenure_days = (now - d.joined_date).days
    trust_score, components, reasons = compute_driver_trust_score(
        rating=d.rating,
        cancel_rate=d.cancel_rate,
        tenure_days=tenure_days,
    )
    return {
        "id": d.id,
        "name": d.name,
        "city": d.city.name if d.city else "",
        "vehicle_type": d.vehicle_type,
        "rating": round(d.rating, 2),
        "cancel_rate": round(d.cancel_rate, 4),
        "tenure_days": tenure_days,
        "trust_score": trust_score,
        "trust_components": components,
        "trust_reasons": reasons,
    }


@router.get("/order/{order_id}/signals")
def order_signals(
    order_id: int,
    _admin: CurrentUser = Depends(require_admin),
) -> dict[str, Any]:
    from fastapi import HTTPException, status as http_status
    now = datetime.utcnow()
    with get_session() as s:
        o: M.Order | None = s.get(M.Order, order_id)
        if not o:
            raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail="Order not found")
        cty = s.get(M.City, o.city_id)
        cust = s.get(M.Customer, o.customer_id)
        drv = s.get(M.Driver, o.driver_id) if o.driver_id is not None else None

        hour = o.created_at.hour
        late_night = hour >= 22 or hour < 5

        customer_payload = _customer_anomaly_payload(s, cust, now) if cust else None
        driver_payload = _driver_trust_payload(s, drv, now) if drv else None

        # Recompute risk contributions for display from current customer signals.
        if customer_payload is not None:
            risk_score, risk_decision, contributions = compute_order_risk_score(
                anomaly_score=float(customer_payload["anomaly_score"]),
                anomaly_flags=customer_payload["anomaly_flags"],
                avg_order_value=float(customer_payload["avg_order_value"]),
                estimated_total=o.total,
                late_night=late_night,
            )
        else:
            risk_score, risk_decision, contributions = (o.risk_score or 0.0, o.risk_decision or "approve", [])

        order_out = {
            "id": o.id,
            "created_at": o.created_at.isoformat(),
            "city": cty.name if cty else "",
            "total": round(o.total, 2),
            "status": o.status,
            "pickup_zone": o.pickup_zone,
            "dropoff_zone": o.dropoff_zone,
            "hour": hour,
            "late_night": late_night,
            "risk_score": round(o.risk_score, 1) if o.risk_score is not None else None,
            "risk_decision": o.risk_decision,
            "customer_id": o.customer_id,
            "merchant_id": o.merchant_id,
            "driver_id": o.driver_id,
        }

        return {
            "order": order_out,
            "customer": customer_payload,
            "driver": driver_payload,
            "risk": {
                "score": o.risk_score if o.risk_score is not None else risk_score,
                "decision": o.risk_decision or risk_decision,
                "contributions": contributions,
            },
        }


@router.get("/driver-trust")
def driver_trust(
    city: str = Query(default="All"),
    _admin: CurrentUser = Depends(require_admin),
) -> dict[str, Any]:
    now = datetime.utcnow()
    cutoff = now - timedelta(days=30)
    bins = 8

    with get_session() as s:
        city_id = _city_id_for_name(s, city)

        # Active driver = ≥1 completed order in last 30d.
        order_filters = [
            M.Order.created_at >= cutoff,
            M.Order.status == "completed",
            M.Order.driver_id.is_not(None),
        ]
        if city_id is not None:
            order_filters.append(M.Order.city_id == city_id)

        active_ids = set(
            r[0] for r in s.execute(
                select(M.Order.driver_id).where(*order_filters).distinct()
            ).all() if r[0] is not None
        )

        if not active_ids:
            return {
                "active_city": city,
                "histogram": [
                    {"lo": round(i * 12.5, 2), "hi": round((i + 1) * 12.5, 2), "count": 0}
                    for i in range(bins)
                ],
                "bottom_10": [],
                "p10": 0.0, "p50": 0.0, "p90": 0.0,
                "active_drivers": 0,
            }

        drivers = s.execute(
            select(M.Driver).where(M.Driver.id.in_(active_ids))
        ).scalars().all()

        scored: list[tuple[float, M.Driver]] = []
        for d in drivers:
            tenure_days = (now - d.joined_date).days
            trust, _components, _reasons = compute_driver_trust_score(
                rating=d.rating,
                cancel_rate=d.cancel_rate,
                tenure_days=tenure_days,
            )
            scored.append((trust, d))

        scores_sorted = sorted(s_ for s_, _ in scored)

        def pct(arr: list[float], q: float) -> float:
            if not arr:
                return 0.0
            idx = max(0, min(len(arr) - 1, int(round((len(arr) - 1) * q))))
            return round(arr[idx], 1)

        # 8 bins across [0, 100].
        width = 100.0 / bins
        bin_counts = [0] * bins
        for sc in scores_sorted:
            i = int(sc / width)
            if i >= bins:
                i = bins - 1
            if i < 0:
                i = 0
            bin_counts[i] += 1
        histogram = [
            {"lo": round(i * width, 2), "hi": round((i + 1) * width, 2), "count": c}
            for i, c in enumerate(bin_counts)
        ]

        # Bottom 10 by trust score.
        scored.sort(key=lambda t: t[0])
        bottom = scored[:10]
        bottom_out = []
        for trust, d in bottom:
            tenure_days = (now - d.joined_date).days
            bottom_out.append({
                "id": d.id,
                "name": d.name,
                "city": d.city.name if d.city else "",
                "vehicle_type": d.vehicle_type,
                "rating": round(d.rating, 2),
                "cancel_rate": round(d.cancel_rate, 4),
                "tenure_days": tenure_days,
                "trust_score": trust,
            })

        return {
            "active_city": city,
            "histogram": histogram,
            "bottom_10": bottom_out,
            "p10": pct(scores_sorted, 0.10),
            "p50": pct(scores_sorted, 0.50),
            "p90": pct(scores_sorted, 0.90),
            "active_drivers": len(scores_sorted),
        }


@router.get("/customer-anomalies")
def customer_anomalies(
    city: str = Query(default="All"),
    limit: int = Query(default=10, ge=1, le=100),
    days: int = Query(default=14),
    _admin: CurrentUser = Depends(require_admin),
) -> dict[str, Any]:
    now = datetime.utcnow()
    activity_cutoff = now - timedelta(days=days)
    recent_cutoff = now - timedelta(days=7)

    with get_session() as s:
        city_id = _city_id_for_name(s, city)

        # Restrict to customers with activity in last `days` days for performance.
        active_filters = [M.Order.created_at >= activity_cutoff]
        if city_id is not None:
            active_filters.append(M.Order.city_id == city_id)
        active_cust_ids = set(
            r[0] for r in s.execute(
                select(M.Order.customer_id).where(*active_filters).distinct()
            ).all() if r[0] is not None
        )

        cust_filters = []
        if city_id is not None:
            cust_filters.append(M.Customer.city_id == city_id)
        if active_cust_ids:
            cust_filters.append(M.Customer.id.in_(active_cust_ids))
        else:
            return {"active_city": city, "top_anomalies": []}

        customers = s.execute(
            select(M.Customer).where(*cust_filters)
        ).scalars().all()

        # Pull all orders for these customers in one round-trip.
        cust_ids_list = [c.id for c in customers]
        orders_by_cust: dict[int, list[M.Order]] = defaultdict(list)
        if cust_ids_list:
            for o in s.execute(
                select(M.Order).where(M.Order.customer_id.in_(cust_ids_list))
            ).scalars().all():
                orders_by_cust[o.customer_id].append(o)

        rows: list[dict[str, Any]] = []
        for c in customers:
            tenure_days = (now - c.signup_date).days
            all_orders = orders_by_cust.get(c.id, [])
            recent_orders = [o for o in all_orders if o.created_at >= recent_cutoff]
            recent_n = len(recent_orders)
            recent_cancels = sum(1 for o in recent_orders if o.status == "cancelled")
            avg_total = (sum(o.total for o in all_orders) / len(all_orders)) if all_orders else 0.0

            score, flags = compute_customer_anomaly_score(
                tenure_days=tenure_days,
                recent_orders_7d=recent_n,
                recent_cancels_7d=recent_cancels,
            )
            if score <= 0:
                continue
            rows.append({
                "id": c.id,
                "name": c.name,
                "city": c.city.name if c.city else "",
                "tenure_days": tenure_days,
                "recent_orders_7d": recent_n,
                "recent_cancels_7d": recent_cancels,
                "avg_order_value": round(avg_total, 2),
                "anomaly_score": score,
                "flags": flags,
            })

        rows.sort(key=lambda r: r["anomaly_score"], reverse=True)
        return {
            "active_city": city,
            "top_anomalies": rows[:limit],
        }
