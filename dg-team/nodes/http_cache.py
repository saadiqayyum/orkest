"""
HTTP fetch utilities with optional caching.

Provides two fetch functions:
- fetch(): Simple HTTP fetch, no caching
- fetch_with_cache(): HTTP fetch with database-backed caching and TTL in hours

Caching decision is made at the call site, keeping it close to the network call.
"""
import httpx
import structlog
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple
from sqlalchemy import select, text

from shared.database import get_db_session

logger = structlog.get_logger()

# Default user agent
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; SubstackAnalyzer/1.0)"


async def fetch(
    url: str,
    client: Optional[httpx.AsyncClient] = None,
) -> str:
    """
    Simple HTTP fetch with no caching.

    Use this for data that changes frequently (e.g., post listings, APIs).

    Args:
        url: URL to fetch
        client: Optional httpx client (creates one if not provided)

    Returns:
        Response content as string
    """
    close_client = False
    if client is None:
        client = httpx.AsyncClient(
            timeout=30,
            follow_redirects=True,
            headers={"User-Agent": DEFAULT_USER_AGENT}
        )
        close_client = True

    try:
        response = await client.get(url)
        if response.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {response.status_code}: {response.reason_phrase}",
                request=response.request,
                response=response,
            )
        logger.debug("fetched", url=url)
        return response.text

    finally:
        if close_client:
            await client.aclose()


async def fetch_with_cache(
    url: str,
    cache_ttl_hours: int = 168,  # 7 days default
    client: Optional[httpx.AsyncClient] = None,
    force_refresh: bool = False,
) -> Tuple[str, bool]:
    """
    HTTP fetch with database-backed caching.

    Use this for content that rarely changes (e.g., individual post content).

    Args:
        url: URL to fetch
        cache_ttl_hours: Cache TTL in hours (default 168 = 7 days, 0 = no expiry)
        client: Optional httpx client (creates one if not provided)
        force_refresh: If True, ignore cache and fetch fresh

    Returns:
        Tuple of (content, from_cache) where from_cache indicates if it was a cache hit
    """
    # Check cache first (unless force refresh)
    if not force_refresh:
        cached = await _get_cached(url)
        if cached:
            return cached, True

    # Fetch fresh
    content = await fetch(url, client)

    # Cache the response
    await _set_cached(url, content, cache_ttl_hours)

    logger.info("fetched_and_cached", url=url, ttl_hours=cache_ttl_hours)
    return content, False


# -----------------------------------------------------------------------------
# Internal cache helpers
# -----------------------------------------------------------------------------

async def _get_cached(url: str) -> Optional[str]:
    """Get cached content for a URL if it exists and hasn't expired."""
    async with get_db_session() as db:
        result = await db.execute(
            text("""
                SELECT content FROM http_cache
                WHERE url = :url
                AND (expires_at IS NULL OR expires_at > NOW())
            """),
            {"url": url}
        )
        row = result.fetchone()
        if row:
            logger.debug("cache_hit", url=url)
            return row[0]
        return None


async def _set_cached(
    url: str,
    content: str,
    cache_ttl_hours: int = 168,
) -> None:
    """Store content in cache with TTL in hours."""
    expires_at = None
    if cache_ttl_hours > 0:
        expires_at = datetime.now(timezone.utc) + timedelta(hours=cache_ttl_hours)

    async with get_db_session() as db:
        await db.execute(
            text("""
                INSERT INTO http_cache (url, content, status_code, content_type, fetched_at, expires_at)
                VALUES (:url, :content, 200, 'text/html', NOW(), :expires_at)
                ON CONFLICT (url) DO UPDATE SET
                    content = EXCLUDED.content,
                    status_code = EXCLUDED.status_code,
                    content_type = EXCLUDED.content_type,
                    fetched_at = NOW(),
                    expires_at = EXCLUDED.expires_at
            """),
            {
                "url": url,
                "content": content,
                "expires_at": expires_at,
            }
        )
        await db.commit()
        logger.debug("cache_set", url=url, expires_at=expires_at)


# -----------------------------------------------------------------------------
# Legacy API (deprecated, use fetch() or fetch_with_cache() instead)
# -----------------------------------------------------------------------------

async def cached_fetch(
    url: str,
    client: Optional[httpx.AsyncClient] = None,
    cache_days: int = 7,
    force_refresh: bool = False,
) -> Tuple[str, bool]:
    """
    DEPRECATED: Use fetch_with_cache() instead.

    Kept for backwards compatibility with existing code.
    """
    return await fetch_with_cache(
        url=url,
        cache_ttl_hours=cache_days * 24,
        client=client,
        force_refresh=force_refresh,
    )


async def get_cached(url: str) -> Optional[str]:
    """DEPRECATED: Internal function, use fetch_with_cache() instead."""
    return await _get_cached(url)


async def set_cached(
    url: str,
    content: str,
    status_code: int = 200,
    content_type: str = "text/html",
    cache_days: int = 7,
) -> None:
    """DEPRECATED: Internal function, use fetch_with_cache() instead."""
    await _set_cached(url, content, cache_days * 24)


async def clear_expired_cache() -> int:
    """
    Remove expired cache entries.

    Returns:
        Number of entries removed
    """
    async with get_db_session() as db:
        result = await db.execute(
            text("DELETE FROM http_cache WHERE expires_at IS NOT NULL AND expires_at < NOW()")
        )
        await db.commit()
        count = result.rowcount
        if count > 0:
            logger.info("cache_cleared", removed=count)
        return count


async def get_cache_stats() -> dict:
    """Get cache statistics."""
    async with get_db_session() as db:
        result = await db.execute(
            text("""
                SELECT
                    COUNT(*) as total_entries,
                    COUNT(CASE WHEN expires_at IS NULL OR expires_at > NOW() THEN 1 END) as valid_entries,
                    COUNT(CASE WHEN expires_at IS NOT NULL AND expires_at < NOW() THEN 1 END) as expired_entries,
                    pg_size_pretty(SUM(LENGTH(content))::bigint) as total_size
                FROM http_cache
            """)
        )
        row = result.fetchone()
        return {
            "total_entries": row[0],
            "valid_entries": row[1],
            "expired_entries": row[2],
            "total_size": row[3],
        }
