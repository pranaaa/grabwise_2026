"""Auth endpoints + cookie-based session helper.

Cookie payload is the auth_user.id, signed with itsdangerous (HMAC).
Login accepts either username or email. Successful logins are tracked
on the user (last_login_at, last_login_ip) and in the login_attempts
audit table. Failed logins increment failed_login_count for forensics.

Phase 1 of the user persistence overhaul:
  - Login by username OR email
  - Audit columns updated on every attempt (last_login_at, failed_login_count)
  - LoginAttempt rows written for both success + failure
  - Register endpoint creates the linked persona + AuthUser in one transaction
"""
from __future__ import annotations
import re
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Response, Request, Depends, status
from pydantic import BaseModel, EmailStr, Field
import bcrypt
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from sqlalchemy import select, or_

from backend.db.database import get_session
from backend.db import models as M

# Hackathon default — overridable via env in production.
SESSION_SECRET = "grabwise-hackathon-secret-please-rotate"
SESSION_COOKIE = "grabwise_session"
SESSION_MAX_AGE = 60 * 60 * 24 * 7  # 1 week

serializer = URLSafeTimedSerializer(SESSION_SECRET, salt="grabwise-session")


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8")[:72], hashed.encode("utf-8"))
    except Exception:
        return False


router = APIRouter(prefix="/api/auth", tags=["auth"])


# ============================================================================
# Pydantic schemas
# ============================================================================
class LoginBody(BaseModel):
    username: str  # accepts either username or email
    password: str


class RegisterBody(BaseModel):
    """Public registration payload.

    Required for every role: full_name, email, password, role, city.
    Role-specific extras carried in optional fields below — ignored when not
    relevant. We don't allow role='admin' via this endpoint (security).
    """
    full_name: str = Field(..., min_length=2, max_length=160)
    email: EmailStr
    username: str | None = Field(None, max_length=80)
    password: str = Field(..., min_length=6, max_length=128)
    phone: str | None = Field(None, max_length=40)
    role: str = Field(..., pattern="^(driver|customer|merchant)$")
    city: str = Field(..., max_length=80)

    # Customer-specific
    dietary_prefs: list[str] | None = None  # ["vegetarian", "halal", ...]

    # Driver-specific
    vehicle_type: str | None = Field(None, pattern="^(bike|car)$")

    # Merchant-specific
    cuisine: str | None = None
    avg_prep_min: int | None = Field(None, ge=5, le=120)
    zone: str | None = None


class CurrentUser(BaseModel):
    id: int                   # persona id (driver_id / customer_id / merchant_id) or auth_user.id for admin
    auth_user_id: int
    username: str
    email: str
    role: str
    display_name: str
    full_name: str
    avatar_url: str | None = None
    phone: str | None = None
    city: str | None = None
    extra: str | None = None
    last_login_at: str | None = None


# ============================================================================
# Helpers
# ============================================================================
def _resolve_display(s, user: M.AuthUser) -> tuple[str, Optional[str], Optional[str], int]:
    """Return (display_name, city, extra, persona_id) for the linked entity.

    persona_id is the id used by the agents (driver_id / customer_id / merchant_id).
    For admin, returns the auth_user.id as persona_id (agents bypass it for admins).
    """
    if user.role == "driver" and user.driver_id:
        d = s.get(M.Driver, user.driver_id)
        if d:
            return d.name, d.city.name, f"{d.vehicle_type} · ⭐ {d.rating}", d.id
    if user.role == "customer" and user.customer_id:
        c = s.get(M.Customer, user.customer_id)
        if c:
            extra = ", ".join(c.dietary_prefs) if c.dietary_prefs else "no restrictions"
            return c.name, c.city.name, extra, c.id
    if user.role == "merchant" and user.merchant_id:
        m = s.get(M.Merchant, user.merchant_id)
        if m:
            return m.name, m.city.name, f"{m.cuisine} · ⭐ {m.rating}", m.id
    return ("Admin" if user.role == "admin" else user.username), None, None, user.id


def _set_session_cookie(response: Response, auth_user_id: int) -> None:
    token = serializer.dumps({"auth_user_id": auth_user_id})
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=False,  # hackathon localhost
    )


def _read_session_cookie(request: Request) -> Optional[int]:
    raw = request.cookies.get(SESSION_COOKIE)
    if not raw:
        return None
    try:
        data = serializer.loads(raw, max_age=SESSION_MAX_AGE)
        return int(data.get("auth_user_id"))
    except (BadSignature, SignatureExpired, ValueError, TypeError):
        return None


def _client_ip(request: Request) -> str | None:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else None


def _user_agent(request: Request) -> str | None:
    ua = request.headers.get("user-agent")
    return ua[:255] if ua else None


def _record_attempt(
    s, identifier: str, success: bool, request: Request, error_reason: str | None = None
) -> None:
    """Audit one login attempt to the login_attempts table."""
    s.add(M.LoginAttempt(
        identifier=identifier[:160],
        success=success,
        error_reason=error_reason,
        ip=_client_ip(request),
        user_agent=_user_agent(request),
        created_at=datetime.utcnow(),
    ))


def _slugify(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 ]+", "", name).strip().lower()
    parts = [p for p in cleaned.split() if p]
    return ".".join(parts[:3]) if parts else "user"


def _to_currentuser(s, user: M.AuthUser) -> CurrentUser:
    display_name, city, extra, persona_id = _resolve_display(s, user)
    return CurrentUser(
        id=persona_id,
        auth_user_id=user.id,
        username=user.username,
        email=user.email,
        role=user.role,
        display_name=display_name,
        full_name=user.full_name,
        avatar_url=user.avatar_url,
        phone=user.phone,
        city=city,
        extra=extra,
        last_login_at=user.last_login_at.isoformat() if user.last_login_at else None,
    )


def get_current_user(request: Request) -> CurrentUser:
    auth_user_id = _read_session_cookie(request)
    if not auth_user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    with get_session() as s:
        user = s.get(M.AuthUser, auth_user_id)
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        if not user.is_active:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")
        return _to_currentuser(s, user)


def require_admin(current: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if current.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin only")
    return current


# ============================================================================
# Endpoints
# ============================================================================
@router.post("/login")
def login(body: LoginBody, response: Response, request: Request):
    """Login with username or email. Tracks audit fields, logs every attempt."""
    identifier = body.username.strip()
    with get_session() as s:
        # Match by username OR email (case-insensitive on email)
        user = s.execute(
            select(M.AuthUser).where(
                or_(
                    M.AuthUser.username == identifier,
                    M.AuthUser.email == identifier.lower(),
                )
            )
        ).scalar_one_or_none()

        if not user:
            _record_attempt(s, identifier, False, request, "user_not_found")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

        if not user.is_active:
            _record_attempt(s, identifier, False, request, "account_disabled")
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")

        if not verify_password(body.password, user.password_hash):
            user.failed_login_count = (user.failed_login_count or 0) + 1
            _record_attempt(s, identifier, False, request, "wrong_password")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

        # Success — update audit fields
        user.last_login_at = datetime.utcnow()
        user.last_login_ip = _client_ip(request)
        user.failed_login_count = 0
        _record_attempt(s, identifier, True, request)
        _set_session_cookie(response, user.id)
        return _to_currentuser(s, user)


@router.post("/register", status_code=201)
def register(body: RegisterBody, response: Response, request: Request):
    """Self-service registration for customer / driver / merchant roles.

    Creates the linked persona row (Driver / Customer / Merchant) AND the
    AuthUser in a single transaction. Auto-logs in on success.
    """
    role = body.role
    with get_session() as s:
        # ---- Validate uniqueness ----
        email_norm = body.email.lower()
        existing_email = s.execute(
            select(M.AuthUser).where(M.AuthUser.email == email_norm)
        ).scalar_one_or_none()
        if existing_email:
            raise HTTPException(status_code=409, detail="An account with this email already exists.")

        username = (body.username or _slugify(body.full_name)).lower()
        # If chosen username conflicts, append a numeric suffix
        n = 1
        original_username = username
        while s.execute(select(M.AuthUser).where(M.AuthUser.username == username)).scalar_one_or_none():
            n += 1
            username = f"{original_username}{n}"

        # ---- Look up city ----
        city = s.scalar(select(M.City).where(M.City.name == body.city))
        if not city:
            raise HTTPException(status_code=400, detail=f"Unknown city: {body.city!r}")

        now = datetime.utcnow()
        avatar = (
            f"https://api.dicebear.com/7.x/initials/svg?seed="
            f"{re.sub(r'[^A-Za-z0-9]+', '', body.full_name) or 'user'}"
            f"&backgroundColor=00B14F&textColor=ffffff"
        )

        # ---- Create the linked persona row ----
        driver_id = customer_id = merchant_id = None

        if role == "driver":
            default_home_zone = city.zones[0] if city.zones else "Downtown"
            d = M.Driver(
                name=body.full_name,
                phone=body.phone or "",
                city_id=city.id,
                vehicle_type=body.vehicle_type or "bike",
                rating=4.6,                  # baseline for new drivers
                cancel_rate=0.05,
                joined_date=now,
                is_active=True,
                behavior_persona="steady-allrounder",
                home_zone=default_home_zone,
            )
            s.add(d); s.flush()
            driver_id = d.id

            # Sensible defaults so the daily planner works immediately
            # on first chat after register. Driver can edit these later.
            s.add(M.DriverPreferences(
                driver_id=d.id,
                preferred_zones=[default_home_zone],
                blackout_hours=[],
                weekly_target_sgd=500.0,
                notify_plan_changes=True,
            ))
            # Default schedule: Mon-Fri, 9am-6pm
            for dow in range(5):
                s.add(M.DriverSchedule(
                    driver_id=d.id,
                    day_of_week=dow,
                    start_hour=9,
                    end_hour=18,
                ))

        elif role == "customer":
            c = M.Customer(
                name=body.full_name,
                city_id=city.id,
                dietary_prefs=body.dietary_prefs or [],
                signup_date=now,
                behavior_persona="new-user",
            )
            s.add(c); s.flush()
            customer_id = c.id

        elif role == "merchant":
            m = M.Merchant(
                name=f"{body.full_name.split(' ')[0]}'s Kitchen" if not body.cuisine else body.full_name,
                city_id=city.id,
                cuisine=body.cuisine or "Local",
                rating=4.4,
                avg_prep_min=body.avg_prep_min or 15,
                zone=body.zone or city.zones[0],
                behavior_persona="rising-star",
            )
            s.add(m); s.flush()
            merchant_id = m.id
        else:
            raise HTTPException(status_code=400, detail=f"Invalid role: {role!r}")

        # ---- Create the AuthUser ----
        user = M.AuthUser(
            email=email_norm,
            username=username,
            phone=body.phone,
            password_hash=hash_password(body.password),
            full_name=body.full_name,
            avatar_url=avatar,
            role=role,
            driver_id=driver_id,
            customer_id=customer_id,
            merchant_id=merchant_id,
            is_active=True,
            is_verified=False,           # email/phone verification not implemented yet
            created_at=now,
            updated_at=now,
        )
        s.add(user); s.flush()

        # ---- Auto-login ----
        user.last_login_at = now
        user.last_login_ip = _client_ip(request)
        _record_attempt(s, email_norm, True, request)
        _set_session_cookie(response, user.id)
        return _to_currentuser(s, user)


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True}


@router.get("/me", response_model=CurrentUser)
def me(current: CurrentUser = Depends(get_current_user)):
    return current
