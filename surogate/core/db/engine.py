"""Database engine and async session management."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from surogate.core.db.base import Base

_engine: AsyncEngine | None = None
_async_session: async_sessionmaker[AsyncSession] | None = None


def init_engine(
    url: str = "sqlite+aiosqlite:///surogate.db", **kwargs
) -> AsyncEngine:
    global _engine, _async_session
    _engine = create_async_engine(url, **kwargs)
    _async_session = async_sessionmaker(
        _engine, class_=AsyncSession, expire_on_commit=False
    )
    return _engine


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise RuntimeError("Call init_engine() first")
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _async_session is None:
        raise RuntimeError("Call init_engine() first")
    return _async_session


async def get_session():
    """FastAPI dependency that yields an async session."""
    factory = get_session_factory()
    async with factory() as session:
        yield session


async def create_all_tables() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def run_migrations(db_url: str | None = None) -> None:
    """Run Alembic migrations (upgrade to head).

    Uses the project's alembic.ini. Optionally overrides the DB URL.
    """
    ini_path = Path(__file__).resolve().parents[3] / "alembic.ini"
    alembic_cfg = Config(str(ini_path))
    if db_url:
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)
    command.upgrade(alembic_cfg, "head")

