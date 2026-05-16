"""Tools the Fraud & Risk Agent calls.

Pillars from the deck:
  • Driver Trust Scoring — score using safety feedback, reliability, complaint history.
  • Trusted Late-Night Matching — prioritize highly-rated, reliable drivers.
  • Customer Anomaly Detection — flag unusual customer behaviour (orders, payments, devices).

Scores are deterministic, derived from the seeded data so the demo is reproducible.
We use simple, interpretable formulas rather than ML — the agent's value is the
narrative it builds around the numbers.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select, func, and_
from langchain_core.tools import tool

from backend.db.database import get_session
from backend.db import models as M
from backend.tools._risk_math import (
    compute_customer_anomaly_score,
    compute_order_risk_score,
    compute_driver_trust_score,
    _clamp,
)


# --------------------------- 1. Driver trust scoring -------------------------
@tool
def score_driver_trust(driver_id: int) -> dict[str, Any]:
    """Compute a 0-100 trust score for a driver.

    Higher = more trusted. Combines rating, cancel rate, and tenure into a
    single readable score with the contributing reasons.

    Args:
        driver_id: Driver's numeric ID.
    """
    with get_session() as s:
        d: M.Driver | None = s.get(M.Driver, driver_id)
        if not d:
            return {"error": f"driver {driver_id} not found"}

        tenure_days = (datetime.utcnow() - d.joined_date).days

        total, components, reasons = compute_driver_trust_score(
            rating=d.rating,
            cancel_rate=d.cancel_rate,
            tenure_days=tenure_days,
        )

        return {
            "driver_id": driver_id,
            "name": d.name,
            "city": d.city.name,
            "trust_score": total,
            "components": components,
            "reasons": reasons,
        }


# --------------------------- 2. Customer anomaly detection -------------------
@tool
def score_customer_anomaly(customer_id: int) -> dict[str, Any]:
    """Compute a 0-100 anomaly score for a customer (higher = more suspicious).

    Looks at recent order velocity, cancel ratio, account tenure, and avg order
    value to flag unusual behaviour.

    Args:
        customer_id: Customer's numeric ID.
    """
    with get_session() as s:
        c: M.Customer | None = s.get(M.Customer, customer_id)
        if not c:
            return {"error": f"customer {customer_id} not found"}

        tenure_days = (datetime.utcnow() - c.signup_date).days

        # Recent activity (7 days)
        recent_cutoff = datetime.utcnow() - timedelta(days=7)
        recent_orders = s.execute(
            select(M.Order).where(
                M.Order.customer_id == customer_id,
                M.Order.created_at >= recent_cutoff,
            )
        ).scalars().all()
        recent_n = len(recent_orders)
        recent_cancels = sum(1 for o in recent_orders if o.status == "cancelled")

        # All-time activity for baseline
        all_orders = s.execute(
            select(M.Order).where(M.Order.customer_id == customer_id)
        ).scalars().all()
        avg_total = (sum(o.total for o in all_orders) / len(all_orders)) if all_orders else 0.0

        anomaly_score, flags = compute_customer_anomaly_score(
            tenure_days=tenure_days,
            recent_orders_7d=recent_n,
            recent_cancels_7d=recent_cancels,
        )

        return {
            "customer_id": customer_id,
            "name": c.name,
            "anomaly_score": anomaly_score,
            "tenure_days": tenure_days,
            "recent_orders_7d": recent_n,
            "recent_cancels_7d": recent_cancels,
            "avg_order_value": round(avg_total, 2),
            "flags": flags or ["no anomalies detected"],
        }


# --------------------------- 3. Order risk scoring ---------------------------
@tool
def score_order_risk(
    customer_id: int,
    estimated_total: float | None = None,
    late_night: bool = False,
) -> dict[str, Any]:
    """Compute a 0-100 risk score for a *prospective* order (before it's placed).

    Synthesizes customer-anomaly signals + amount-based heuristics + late-night
    flag. Lower = safer to place. Use this in the cross-agent ordering chain.

    Args:
        customer_id: Customer placing the order.
        estimated_total: Optional, expected order amount. >2× the customer's
            historical average bumps the risk score.
        late_night: True if the order is being placed at late-night hours
            (typically 22:00–05:00). Adds caution.
    """
    anomaly = score_customer_anomaly.invoke({"customer_id": customer_id})
    if "error" in anomaly:
        return anomaly

    risk_score, decision, contributions = compute_order_risk_score(
        anomaly_score=float(anomaly["anomaly_score"]),
        anomaly_flags=anomaly["flags"],
        avg_order_value=float(anomaly["avg_order_value"]),
        estimated_total=estimated_total,
        late_night=late_night,
    )

    return {
        "customer_id": customer_id,
        "risk_score": risk_score,
        "decision": decision,
        "contributions": contributions,
    }


# --------------------------- 4. Transaction signals --------------------------
@tool
def get_transaction_signals(order_id: int) -> dict[str, Any]:
    """Pull risk-relevant signals for an existing order.

    Use this when the user references a specific past order_id (e.g. for a
    chargeback investigation).

    Args:
        order_id: The order/transaction id.
    """
    with get_session() as s:
        o: M.Order | None = s.get(M.Order, order_id)
        if not o:
            return {"error": f"order {order_id} not found"}

        return {
            "order_id": o.id,
            "customer_id": o.customer_id,
            "merchant_id": o.merchant_id,
            "driver_id": o.driver_id,
            "city_id": o.city_id,
            "total": o.total,
            "driver_earning": o.driver_earning,
            "status": o.status,
            "created_at": o.created_at.isoformat(),
            "pickup_zone": o.pickup_zone,
            "dropoff_zone": o.dropoff_zone,
        }


FRAUD_TOOLS = [
    score_driver_trust,
    score_customer_anomaly,
    score_order_risk,
    get_transaction_signals,
]
