"""
backend/database.py
===================
SQLAlchemy 2.x database layer.

Provides:
  - engine        : shared Engine instance connected to SQLite
  - SessionLocal  : session factory for request-scoped sessions
  - get_db()      : FastAPI dependency that yields a Session and
                    guarantees cleanup even on exceptions
  - create_tables(): idempotent DDL function called at startup
"""

from __future__ import annotations

import logging
from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import settings

logger = logging.getLogger(__name__)

# ── Engine ────────────────────────────────────────────────────────────────────

# SQLite needs connect_args={"check_same_thread": False} to run inside FastAPI safely.
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=settings.DEBUG,    # log all SQL when DEBUG=true
)

# ── Session factory ───────────────────────────────────────────────────────────

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,  # keep ORM objects usable after commit without re-querying
)

# ── Declarative base ──────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


# ── FastAPI dependency ────────────────────────────────────────────────────────

def get_db() -> Generator[Session, None, None]:
    """
    Yield a database session for the duration of a single HTTP request.

    Usage in a route:
        @router.get("/items")
        def read_items(db: Session = Depends(get_db)):
            ...

    The session is always closed in the finally block regardless of
    whether the request raised an exception.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ── Table creation ────────────────────────────────────────────────────────────

def create_tables() -> None:
    """
    Create all tables defined in the ORM models if they do not already exist.

    This is idempotent — safe to call on every startup.
    Import models here (not at module level) to avoid circular imports;
    the import side-effect registers the models with Base.metadata.
    """
    import backend.models  # no
    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)
        logger.info("Database tables verified / created successfully.")
    except Exception as exc:
        logger.exception("Failed to create database tables: %s", exc)
        raise


def check_connection() -> bool:
    """
    Return True if the database is reachable, False otherwise.
    Used for the /health endpoint.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("Database connectivity check failed: %s", exc)
        return False