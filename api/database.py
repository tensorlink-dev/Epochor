"""Database helpers for the platform API."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from .config import get_settings


Base = declarative_base()


def _build_engine(database_url: str):
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, future=True, connect_args=connect_args)


engine = _build_engine(get_settings().database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def configure_engine(database_url: str | None = None) -> None:
    """Reconfigure the engine and session factory (useful for tests)."""

    global engine, SessionLocal
    settings = get_settings()
    url = database_url or settings.database_url
    engine = _build_engine(url)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Ensure database tables exist."""

    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""

    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a session."""

    with session_scope() as session:
        yield session


__all__ = ["Base", "engine", "SessionLocal", "init_db", "get_session", "session_scope"]
