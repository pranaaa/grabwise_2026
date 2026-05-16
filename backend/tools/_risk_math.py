"""Internal helpers for risk/anomaly scoring.

Pure functions so the live `score_order_risk` tool and the seed-time precompute
share one formula. Mirror the logic in `fraud_tools.py` exactly.
"""
from __future__ import annotations
from typing import Any


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return round(max(lo, min(hi, x)), 1)


def compute_customer_anomaly_score(
    tenure_days: int,
    recent_orders_7d: int,
    recent_cancels_7d: int,
) -> tuple[float, list[str]]:
    """Returns (anomaly_score, flags). Same formula as the tool."""
    anomaly = 0.0
    flags: list[str] = []
    if tenure_days < 7:
        anomaly += 30
        flags.append(f"new account ({tenure_days}d old)")
    elif tenure_days < 30:
        anomaly += 12
        flags.append(f"young account ({tenure_days}d)")
    if recent_orders_7d >= 15:
        anomaly += 25
        flags.append(f"high recent volume ({recent_orders_7d} orders in 7d)")
    elif recent_orders_7d >= 8:
        anomaly += 10
        flags.append(f"elevated recent volume ({recent_orders_7d} orders in 7d)")
    if recent_orders_7d > 0 and recent_cancels_7d / recent_orders_7d >= 0.30:
        anomaly += 20
        flags.append(f"high recent cancel ratio ({recent_cancels_7d}/{recent_orders_7d})")
    return _clamp(anomaly), flags


def compute_order_risk_score(
    *,
    anomaly_score: float,
    anomaly_flags: list[str],
    avg_order_value: float,
    estimated_total: float | None,
    late_night: bool,
) -> tuple[float, str, list[dict[str, Any]]]:
    """Returns (risk_score, decision, contributions)."""
    risk = float(anomaly_score)
    contributions: list[dict[str, Any]] = [{
        "factor": "customer_anomaly",
        "delta": risk,
        "note": ", ".join(anomaly_flags) if anomaly_flags else "no anomalies detected",
    }]

    if estimated_total is not None and avg_order_value > 0:
        ratio = estimated_total / avg_order_value
        if ratio >= 2.0:
            risk += 15
            contributions.append({
                "factor": "abnormally_high_amount",
                "delta": 15,
                "note": f"order ${estimated_total:.2f} is {ratio:.1f}× the customer's avg",
            })
        elif ratio >= 1.5:
            risk += 5
            contributions.append({
                "factor": "above_avg_amount",
                "delta": 5,
                "note": f"order is {ratio:.1f}× the customer's avg",
            })

    if late_night:
        risk += 5
        contributions.append({
            "factor": "late_night",
            "delta": 5,
            "note": "late-night orders carry mild additional caution",
        })

    final = _clamp(risk)
    decision = "approve" if final < 35 else ("review" if final < 65 else "block")
    return final, decision, contributions


def compute_driver_trust_score(
    *,
    rating: float,
    cancel_rate: float,
    tenure_days: int,
) -> tuple[float, dict[str, float], list[str]]:
    """Returns (trust_score, components, reasons). Mirrors `score_driver_trust`."""
    rating_pts = max(0.0, (rating - 4.0) * 50)
    cancel_pts = max(0.0, 30.0 * (1.0 - min(cancel_rate / 0.25, 1.0)))
    tenure_pts = min(tenure_days / 730.0, 1.0) * 20.0
    total = _clamp(rating_pts + cancel_pts + tenure_pts)

    reasons: list[str] = []
    if rating >= 4.8:
        reasons.append(f"high rating ({rating}★)")
    elif rating < 4.3:
        reasons.append(f"below-average rating ({rating}★)")
    if cancel_rate <= 0.03:
        reasons.append(f"very low cancel rate ({cancel_rate * 100:.1f}%)")
    elif cancel_rate >= 0.10:
        reasons.append(f"elevated cancel rate ({cancel_rate * 100:.1f}%)")
    if tenure_days >= 365:
        reasons.append(f"established driver ({tenure_days // 365}+ year tenure)")
    elif tenure_days < 60:
        reasons.append(f"new driver ({tenure_days}d tenure)")

    components = {
        "rating_pts": round(rating_pts, 1),
        "cancel_rate_pts": round(cancel_pts, 1),
        "tenure_pts": round(tenure_pts, 1),
    }
    return total, components, (reasons or ["typical profile"])
