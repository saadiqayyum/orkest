"""
Database connection and session management for dg-team.

This module provides async database access for the mirroring engine's
domain-specific tables. Connection string is loaded from team's .env file.

Constitution Note: This is the team's own database connection, separate
from the engine's database. The engine tracks workflow runs; this tracks
publication content.

IMPORTANT: Database connection is lazily initialized to support dynamic
environment variable injection. The engine/executor injects vault env vars
into os.environ right before running workflows, so we must NOT read env
vars at module import time.
"""
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.pool import AsyncAdaptedQueuePool
import structlog

from .models import Base

logger = structlog.get_logger()

# Lazy initialization - engine and session maker created on first use
# We track the URL used to create the engine so we can recreate it if the URL changes
# (e.g., different projects with different database URLs)
_engine = None
_async_session_maker = None
_current_database_url = None


def _get_database_url() -> str:
    """Get and validate database URL from environment."""
    raw_url = os.environ.get("DG_TEAM_DATABASE_URL") or os.environ.get("DATABASE_URL")

    if not raw_url:
        raise RuntimeError(
            "DATABASE_URL environment variable is required. "
            "Set it in the project's vault ENV configuration."
        )

    # Ensure we use asyncpg driver (convert postgresql:// to postgresql+asyncpg://)
    if raw_url.startswith("postgresql://"):
        return raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif raw_url.startswith("postgres://"):
        return raw_url.replace("postgres://", "postgresql+asyncpg://", 1)
    else:
        return raw_url


def _get_engine():
    """
    Get or create the database engine (lazy initialization).

    If the database URL has changed since the engine was created,
    dispose the old engine and create a new one. This supports
    multiple projects with different database URLs running in the
    same process.
    """
    global _engine, _async_session_maker, _current_database_url

    database_url = _get_database_url()

    # Check if URL changed - need to recreate engine
    if _engine is not None and _current_database_url != database_url:
        logger.info(
            "database_url_changed_recreating_engine",
            old_url=_current_database_url[:50] + "..." if _current_database_url else None,
            new_url=database_url[:50] + "...",
        )
        # Dispose old engine (closes all connections in pool)
        # Note: This is sync dispose; for async we'd need to handle differently
        _engine.sync_engine.dispose()
        _engine = None
        _async_session_maker = None

    if _engine is None:
        logger.info("initializing_database_engine", url=database_url[:50] + "...")
        _engine = create_async_engine(
            database_url,
            echo=False,
            poolclass=AsyncAdaptedQueuePool,
            pool_size=10,       # Max persistent connections
            max_overflow=20,    # Extra connections when pool is exhausted
            pool_timeout=30,    # Wait up to 30s for a connection
            pool_recycle=3600,  # Recycle connections after 1 hour
        )
        _current_database_url = database_url

    return _engine


def _get_session_maker():
    """
    Get or create the session maker (lazy initialization).

    Note: _get_engine() handles URL changes and resets _async_session_maker
    when needed, so we just need to check if it's None.
    """
    global _async_session_maker
    # Call _get_engine() first to trigger URL change detection
    engine = _get_engine()
    if _async_session_maker is None:
        _async_session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session_maker


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions.

    Usage:
        async with get_db_session() as db:
            result = await db.execute(...)
    """
    session_maker = _get_session_maker()
    session = session_maker()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def init_db():
    """
    Initialize database tables.

    Call this once during setup to create all tables if they don't exist.
    In production, use Alembic migrations instead.
    """
    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_initialized", tables=list(Base.metadata.tables.keys()))


async def check_db_connection() -> bool:
    """
    Check if database is accessible.

    Returns:
        True if connection successful, False otherwise
    """
    try:
        engine = _get_engine()
        async with engine.connect() as conn:
            await conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error("database_connection_failed", error=str(e))
        return False
