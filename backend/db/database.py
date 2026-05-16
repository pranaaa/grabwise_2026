"""SQLAlchemy engine + session factory. Sync mode for hackathon simplicity."""
from __future__ import annotations
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase

from backend.config import settings


class Base(DeclarativeBase):
    pass


# SQLite needs check_same_thread=False to work across threads (FastAPI/agents).
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


@contextmanager
def get_session() -> Session:
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def init_db() -> None:
    """Create all tables. Idempotent."""
    # Import models so SQLAlchemy registers them on Base.metadata.
    from backend.db import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
