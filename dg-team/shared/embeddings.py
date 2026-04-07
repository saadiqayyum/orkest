"""
Embedding generation utilities for user workflows.

This module provides embedding generation using the configured LLM provider.
"""
import sys
from pathlib import Path

# Add backend to path so we can import from it
backend_path = Path(__file__).parent.parent.parent.parent / "backend" / "src"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Re-export embedding utilities from backend
from utils.embeddings import (
    generate_embedding,
    generate_embeddings_batch,
    generate_post_embedding,
)
from config.settings import get_llm_provider

__all__ = [
    "generate_embedding",
    "generate_embeddings_batch",
    "generate_post_embedding",
    "get_llm_provider",
]
