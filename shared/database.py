"""
Database connection and session management for dg-team.

This module provides async database access for the mirroring engine's
domain-specific tables. Connection string is loaded from team's .env file.

Constitution Note: This is the team's own database connection, separate
from the engine's database. The engine tracks workflow runs; this tracks
publication content.
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

# Database URL from environment (team's .env)
# Expected format: postgresql+asyncpg://user:pass@host:port/dbname
_raw_url = os.environ.get(
    "DG_TEAM_DATABASE_URL",
    os.environ.get("DATABASE_URL", "postgresql+asyncpg://orkest:orkest@localhost:5432/dg_team")
)

# Ensure we use asyncpg driver (convert postgresql:// to postgresql+asyncpg://)
if _raw_url.startswith("postgresql://"):
    DATABASE_URL = _raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
elif _raw_url.startswith("postgres://"):
    DATABASE_URL = _raw_url.replace("postgres://", "postgresql+asyncpg://", 1)
else:
    DATABASE_URL = _raw_url

# Create async engine with connection pooling
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    poolclass=AsyncAdaptedQueuePool,
    pool_size=10,       # Max persistent connections
    max_overflow=20,    # Extra connections when pool is exhausted
    pool_timeout=30,    # Wait up to 30s for a connection
    pool_recycle=3600,  # Recycle connections after 1 hour
)

# Session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions.

    Usage:
        async with get_db_session() as db:
            result = await db.execute(...)
    """
    session = async_session_maker()
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
        async with engine.connect() as conn:
            await conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error("database_connection_failed", error=str(e))
        return False
