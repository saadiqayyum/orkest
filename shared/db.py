"""
Database utilities for user workflows.

This module provides access to the user's database.
It imports from the main backend to reuse existing models and connection.
"""
import sys
from pathlib import Path

# Add backend to path so we can import from it
backend_path = Path(__file__).parent.parent.parent.parent / "backend" / "src"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Re-export database components from backend
from db.connection import get_db, async_session_maker
from db.models import Publication, Post, PostEmbedding
from db.queries import (
    get_or_create_publication,
    update_publication_sync_time,
    upsert_post,
    get_posts_by_publication,
    get_posts_without_embeddings,
    upsert_post_embedding,
)

__all__ = [
    "get_db",
    "async_session_maker",
    "Publication",
    "Post",
    "PostEmbedding",
    "get_or_create_publication",
    "update_publication_sync_time",
    "upsert_post",
    "get_posts_by_publication",
    "get_posts_without_embeddings",
    "upsert_post_embedding",
]
