"""Admin-only user listing — useful for the admin to inspect personas."""
from __future__ import annotations
from fastapi import APIRouter, Query, Depends
from sqlalchemy import select

from backend.db.database import get_session
from backend.db import models as M
from backend.api.schemas import UserSummary, UserRole
from backend.api.auth import require_admin, CurrentUser

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("", response_model=list[UserSummary])
def list_users(
    role: UserRole,
    limit: int = Query(default=12, le=50),
    _admin: CurrentUser = Depends(require_admin),
):
    with get_session() as s:
        if role == "driver":
            rows = s.execute(
                select(M.Driver).order_by(M.Driver.rating.desc()).limit(limit)
            ).scalars().all()
            return [
                UserSummary(
                    id=r.id, name=r.name, role="driver", city=r.city.name,
                    extra=f"{r.vehicle_type} · ⭐ {r.rating}",
                ) for r in rows
            ]
        if role == "customer":
            rows = s.execute(select(M.Customer).limit(limit)).scalars().all()
            return [
                UserSummary(
                    id=r.id, name=r.name, role="customer", city=r.city.name,
                    extra=", ".join(r.dietary_prefs) if r.dietary_prefs else "no restrictions",
                ) for r in rows
            ]
        if role == "merchant":
            rows = s.execute(
                select(M.Merchant).order_by(M.Merchant.rating.desc()).limit(limit)
            ).scalars().all()
            return [
                UserSummary(
                    id=r.id, name=r.name, role="merchant", city=r.city.name,
                    extra=f"{r.cuisine} · ⭐ {r.rating}",
                ) for r in rows
            ]
        return []
