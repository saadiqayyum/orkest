#!/usr/bin/env python3
"""
Database setup script for dg-team.

Creates the dg_team database and all required tables.
Uses the same PostgreSQL instance as the Orkest engine.

Usage:
    python scripts/setup_db.py
"""
import asyncio
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine

# Default connection (to postgres db for creating new database)
# Uses same credentials as Orkest engine
POSTGRES_USER = "orkest"
POSTGRES_PASSWORD = "orkest"
POSTGRES_HOST = "localhost"
POSTGRES_PORT = 5432
POSTGRES_URL = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/postgres"
DG_TEAM_DB = "dg_team"
DG_TEAM_URL = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{DG_TEAM_DB}"


async def create_database():
    """Create the dg_team database if it doesn't exist."""
    # Connect to default postgres database
    conn = await asyncpg.connect(
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database="postgres",
    )

    try:
        # Check if database exists
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            DG_TEAM_DB,
        )

        if not exists:
            # Create database
            await conn.execute(f'CREATE DATABASE "{DG_TEAM_DB}"')
            print(f"✓ Created database: {DG_TEAM_DB}")
        else:
            print(f"✓ Database already exists: {DG_TEAM_DB}")

    finally:
        await conn.close()


async def create_extensions():
    """Create required PostgreSQL extensions."""
    conn = await asyncpg.connect(
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=DG_TEAM_DB,
    )

    try:
        # Create pgvector extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        print("✓ Created extension: vector")

    finally:
        await conn.close()


async def create_tables():
    """Create all tables using SQLAlchemy models."""
    from shared.models import Base

    engine = create_async_engine(DG_TEAM_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await engine.dispose()

    print("✓ Created tables:")
    for table_name in Base.metadata.tables.keys():
        print(f"  - {table_name}")


async def create_enum_type():
    """Create the queue_status enum type if not exists."""
    conn = await asyncpg.connect(
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=DG_TEAM_DB,
    )

    try:
        # Check if enum exists
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_type WHERE typname = 'queue_status'"
        )

        if not exists:
            await conn.execute("""
                CREATE TYPE queue_status AS ENUM ('pending', 'drafting', 'review', 'published')
            """)
            print("✓ Created enum type: queue_status")
        else:
            print("✓ Enum type already exists: queue_status")

    finally:
        await conn.close()


async def migrate_target_config():
    """Migrate target_config table to support separate hero/inline image prompts."""
    conn = await asyncpg.connect(
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=DG_TEAM_DB,
    )

    try:
        # Check if target_config table exists
        table_exists = await conn.fetchval(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'target_config'"
        )

        if not table_exists:
            # Create the table with new schema
            await conn.execute("""
                CREATE TABLE target_config (
                    target_handle VARCHAR(100) PRIMARY KEY,
                    article_prompt TEXT,
                    hero_image_prompt TEXT,
                    inline_image_prompt TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            print("✓ Created table: target_config")
        else:
            # Check if new columns exist
            hero_col_exists = await conn.fetchval(
                "SELECT 1 FROM information_schema.columns WHERE table_name = 'target_config' AND column_name = 'hero_image_prompt'"
            )

            if not hero_col_exists:
                # Migrate: rename image_prompt to hero_image_prompt, add inline_image_prompt
                await conn.execute("""
                    ALTER TABLE target_config
                    ADD COLUMN IF NOT EXISTS hero_image_prompt TEXT,
                    ADD COLUMN IF NOT EXISTS inline_image_prompt TEXT
                """)

                # Copy existing image_prompt to hero_image_prompt
                await conn.execute("""
                    UPDATE target_config
                    SET hero_image_prompt = image_prompt
                    WHERE hero_image_prompt IS NULL AND image_prompt IS NOT NULL
                """)

                # Drop old column if it exists
                old_col_exists = await conn.fetchval(
                    "SELECT 1 FROM information_schema.columns WHERE table_name = 'target_config' AND column_name = 'image_prompt'"
                )
                if old_col_exists:
                    await conn.execute("ALTER TABLE target_config DROP COLUMN image_prompt")

                print("✓ Migrated target_config: image_prompt -> hero_image_prompt, added inline_image_prompt")
            else:
                print("✓ Table target_config already has new schema")

    finally:
        await conn.close()


async def create_vector_indexes():
    """Create IVFFlat indexes for vector similarity search."""
    conn = await asyncpg.connect(
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=DG_TEAM_DB,
    )

    try:
        # Index for discovered_posts embeddings
        # Note: IVFFlat requires data in the table to determine list count
        # We use a conservative lists=100 which works well for ~10k-100k rows
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_discovered_posts_embedding
            ON discovered_posts
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
        print("✓ Created vector index: idx_discovered_posts_embedding")

        # Index for discovered_publications embeddings
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_discovered_pubs_embedding
            ON discovered_publications
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
        print("✓ Created vector index: idx_discovered_pubs_embedding")

    except Exception as e:
        # IVFFlat index creation may fail if table is empty
        # This is OK - the index will work without it, just slower
        print(f"⚠ Vector index creation skipped (may need data first): {e}")

    finally:
        await conn.close()


async def main():
    print("=" * 60)
    print("DG-TEAM DATABASE SETUP")
    print("=" * 60)
    print()

    print("1. Creating database...")
    await create_database()
    print()

    print("2. Creating extensions...")
    await create_extensions()
    print()

    print("3. Creating enum types...")
    await create_enum_type()
    print()

    print("4. Creating tables...")
    await create_tables()
    print()

    print("5. Migrating target_config table...")
    await migrate_target_config()
    print()

    print("6. Creating vector indexes...")
    await create_vector_indexes()
    print()

    print("=" * 60)
    print("✓ Setup complete!")
    print()
    print(f"Connection URL: {DG_TEAM_URL}")
    print()
    print("Add to your .env:")
    print(f"  DG_TEAM_DATABASE_URL={DG_TEAM_URL}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
