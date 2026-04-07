"""
Embedding generation node functions for the AI Substack Mirroring Engine.

These workflow nodes handle vector embedding generation:
- generate_embedding: Single text embedding
- batch_embed: Batch embedding for multiple items
- embed_for_dedup: Generate embedding for deduplication check
"""
import os
from typing import List
import structlog
import httpx

from .schemas import (
    GenerateEmbeddingInput, GenerateEmbeddingOutput,
    BatchEmbedInput, BatchEmbedOutput, EmbeddedItem,
)

logger = structlog.get_logger()

# Embedding configuration from environment
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "ollama")  # "openai" or "ollama"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "768"))


async def generate_embedding(
    ctx,
    params: GenerateEmbeddingInput,
) -> GenerateEmbeddingOutput:
    """
    Generate embedding for a single text.
    """
    text = params.text

    ctx.report_input({
        "text_length": len(text) if text else 0,
        "provider": EMBEDDING_PROVIDER,
        "model": EMBEDDING_MODEL,
    })

    if not text or not text.strip():
        ctx.report_output({
            "status": "empty_text",
            "embedding_length": 0,
        })
        return GenerateEmbeddingOutput(status="empty_text")

    try:
        if EMBEDDING_PROVIDER == "openai":
            embedding = await _openai_embed(text)
        elif EMBEDDING_PROVIDER == "ollama":
            embedding = await _ollama_embed(text)
        else:
            ctx.report_output({
                "status": "invalid_provider",
                "provider": EMBEDDING_PROVIDER,
            })
            return GenerateEmbeddingOutput(status="invalid_provider")

        ctx.report_output({
            "status": "success",
            "embedding_length": len(embedding),
            "dimensions": len(embedding),
        })

        return GenerateEmbeddingOutput(
            embedding=embedding,
            dimensions=len(embedding),
            status="success",
        )

    except Exception as e:
        logger.error("embedding_generation_failed", error=str(e))
        ctx.report_output({
            "status": "error",
            "error": str(e),
        })
        return GenerateEmbeddingOutput(status="error")


async def batch_embed(
    ctx,
    params: BatchEmbedInput,
) -> BatchEmbedOutput:
    """
    Generate embeddings for a batch of items.
    """
    items = params.items
    text_field = params.text_field

    ctx.report_input({
        "items_count": len(items),
        "text_field": text_field,
        "provider": EMBEDDING_PROVIDER,
        "model": EMBEDDING_MODEL,
    })

    results = []
    success_count = 0
    failed_count = 0
    failed_items = []

    for item in items:
        # Handle both Pydantic models and dicts (items may be dicts after serialization)
        if isinstance(item, dict):
            text = item.get(text_field, "") or ""
            item_id = item.get("id")
            original_data = item  # Preserve original dict
        elif hasattr(item, text_field):
            text = getattr(item, text_field, None) or ""
            item_id = getattr(item, "id", None)
            # Convert Pydantic model to dict for preservation
            original_data = item.model_dump() if hasattr(item, 'model_dump') else dict(item)
        else:
            text = ""
            item_id = None
            original_data = {}

        if not text:
            logger.warning("empty_text_for_embedding", item_id=item_id)
            failed_count += 1
            failed_items.append({"id": item_id, "reason": "empty_text"})
            continue

        try:
            if EMBEDDING_PROVIDER == "openai":
                embedding = await _openai_embed(text)
            elif EMBEDDING_PROVIDER == "ollama":
                embedding = await _ollama_embed(text)
            else:
                failed_count += 1
                failed_items.append({"id": item_id, "reason": "invalid_provider"})
                continue

            # Merge embedding into original item data
            # This preserves fields like title, summary, rationale, etc.
            result_item = {
                **original_data,
                "id": item_id,
                "embedding": embedding,
                "text": text,
            }
            results.append(result_item)
            success_count += 1

        except Exception as e:
            logger.warning("batch_embed_item_failed", item_id=item_id, error=str(e))
            failed_count += 1
            failed_items.append({"id": item_id, "reason": str(e)})
            continue

    logger.info(
        "batch_embed_complete",
        total=len(items),
        success=success_count,
        failed=failed_count,
    )

    ctx.report_output({
        "embedded_items": [
            {"id": r.get("id"), "text_preview": r.get("text", "")[:100], "embedding_length": len(r.get("embedding", []))}
            for r in results
        ],
        "failed_items": failed_items,
        "success_count": success_count,
        "failed_count": failed_count,
        "status": "success" if success_count > 0 else "all_failed",
    })

    return BatchEmbedOutput(
        items=results,
        success_count=success_count,
        failed_count=failed_count,
        status="success" if success_count > 0 else "all_failed",
    )


async def embed_for_dedup(
    ctx,
    params: GenerateEmbeddingInput,
) -> GenerateEmbeddingOutput:
    """
    Generate embedding specifically for deduplication check.

    This is used before adding new items to check similarity
    against existing items.
    """
    return await generate_embedding(ctx, params)


async def batch_embed_source_items(
    ctx,
    params: BatchEmbedInput,
) -> BatchEmbedOutput:
    """
    Generate embeddings for source items (uses title + content snippet).

    Creates a richer embedding by combining title with content excerpt.
    """
    items = params.items

    ctx.report_input({
        "items_count": len(items),
        "mode": "source_items_combined",
    })

    results = []
    success_count = 0
    failed_count = 0
    failed_items = []

    for item in items:
        # Handle both Pydantic models and dicts
        if isinstance(item, dict):
            title = item.get("title", "") or ""
            content = item.get("content_raw", "") or ""
            item_id = item.get("id")
        else:
            title = getattr(item, "title", "") or ""
            content = getattr(item, "content_raw", "") or ""
            item_id = getattr(item, "id", None)

        # Take first 500 chars of content
        content_snippet = content[:500] if content else ""

        combined_text = f"{title}\n\n{content_snippet}".strip()

        if not combined_text:
            logger.warning("empty_combined_text", item_id=item_id)
            failed_count += 1
            failed_items.append({"id": item_id, "reason": "empty_text"})
            continue

        try:
            if EMBEDDING_PROVIDER == "openai":
                embedding = await _openai_embed(combined_text)
            elif EMBEDDING_PROVIDER == "ollama":
                embedding = await _ollama_embed(combined_text)
            else:
                failed_count += 1
                failed_items.append({"id": item_id, "reason": "invalid_provider"})
                continue

            # Return dict, not Pydantic model (BatchEmbedOutput.items expects dicts)
            results.append({
                "id": item_id,
                "embedding": embedding,
                "text": combined_text,
            })
            success_count += 1

        except Exception as e:
            logger.warning("batch_embed_item_failed", item_id=item_id, error=str(e))
            failed_count += 1
            failed_items.append({"id": item_id, "reason": str(e)})
            continue

    logger.info(
        "batch_embed_source_items_complete",
        total=len(items),
        success=success_count,
        failed=failed_count,
    )

    ctx.report_output({
        "embedded_items": [
            {"id": r.get("id"), "text_preview": r.get("text", "")[:100], "embedding_length": len(r.get("embedding", []))}
            for r in results
        ],
        "failed_items": failed_items,
        "success_count": success_count,
        "failed_count": failed_count,
        "status": "success" if success_count > 0 else "all_failed",
    })

    return BatchEmbedOutput(
        items=results,
        success_count=success_count,
        failed_count=failed_count,
        status="success" if success_count > 0 else "all_failed",
    )


# =============================================================================
# Text Normalization for Embeddings
# =============================================================================

def _normalize_for_embedding(text: str) -> str:
    """
    Normalize text before embedding to improve semantic matching.

    nomic-embed-text is case-sensitive, so we convert to Title Case
    to ensure consistent embeddings regardless of input case.
    This dramatically improves similarity scores for matching concepts.
    """
    if not text:
        return text
    # Convert to title case for consistent embeddings
    # "system design" -> "System Design"
    # "SYSTEM DESIGN" -> "System Design"
    return text.title()


# =============================================================================
# Provider Implementations
# =============================================================================

async def _openai_embed(text: str, normalize: bool = True) -> List[float]:
    """Generate embedding using OpenAI API."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")

    # Normalize text for consistent embeddings
    if normalize:
        text = _normalize_for_embedding(text)

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": EMBEDDING_MODEL,
                "input": text,
                "dimensions": EMBEDDING_DIMENSIONS,
            },
        )
        response.raise_for_status()
        data = response.json()

        return data["data"][0]["embedding"]


async def _ollama_embed(text: str, normalize: bool = True) -> List[float]:
    """Generate embedding using Ollama."""
    # Normalize text for consistent embeddings
    if normalize:
        text = _normalize_for_embedding(text)

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={
                "model": EMBEDDING_MODEL,
                "prompt": text,
            },
        )
        response.raise_for_status()
        data = response.json()

        return data["embedding"]


# =============================================================================
# Unified Posts Embedding Function
# =============================================================================

async def batch_embed_posts(
    ctx,
    params: BatchEmbedInput,
) -> BatchEmbedOutput:
    """
    Generate embeddings for posts from the unified posts table.

    This is an alias for batch_embed_source_items - posts use the same
    schema (id, title, content_raw).
    """
    return await batch_embed_source_items(ctx, params)
