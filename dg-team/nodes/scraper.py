"""
Scraper node functions for the AI Substack Mirroring Engine.

These workflow nodes handle scraping of Substack publications:
- scrape_index: Fetch post index from publications (uses SubstackClient)
- scrape_full_content: Fetch full content for new posts (supports authenticated access for paid content)
- fetch_post_listings: New function using SubstackClient for unified posts table
"""
import structlog
from bs4 import BeautifulSoup
from typing import Optional, Dict, List

from shared.substack_scraper import SubstackScraper, SubstackAuth, fetch_publication_posts
from shared.substack_client import SubstackClient, SubstackPost, create_client, FetchStatus
from .schemas import (
    ScrapeIndexInput, ScrapeIndexOutput, PostMetadata,
    ScrapeFullContentInput, ScrapeFullContentOutput, ScrapedItem,
    ScrapeMetricsBatchInput, ScrapeMetricsBatchOutput, MetricsUpdate,
    PostData,
)
from .http_cache import fetch_with_cache

logger = structlog.get_logger()


async def scrape_index(
    ctx,
    params: ScrapeIndexInput,
) -> ScrapeIndexOutput:
    """
    Fetch post listings from source and target Substack publications.

    Returns post metadata (URL, title, author, engagement metrics) for all posts
    found in the publication feeds. Does NOT fetch full article content - that's
    done later by scrape_full_content for posts that are actually new.

    Output:
    - source_posts: Posts from competitor publications
    - target_posts: Posts from your own publication
    """
    source_handles = params.source_handles
    target_handle = params.target_handle

    # Normalize handles - extract handle if full URL provided, strip whitespace
    source_handles = [_normalize_handle(h) for h in source_handles if h]
    target_handle = _normalize_handle(target_handle) if target_handle else ""

    ctx.report_input({
        "source_handles": source_handles,
        "target_handle": target_handle,
    })

    scraper = SubstackScraper()
    total_handles = len(source_handles) + (1 if target_handle else 0)
    completed = 0
    failed_handles = []

    try:
        all_source_posts = []

        # Scrape all source publications
        for handle in source_handles:
            if not handle:
                logger.warning("empty_source_handle")
                continue

            try:
                metadata, posts = await fetch_publication_posts(handle)

                for post in posts:
                    all_source_posts.append(PostMetadata(
                        url=post.url,
                        title=post.title,
                        author=post.author,
                        publication_handle=handle,
                        likes_count=post.likes_count or 0,
                        comments_count=post.comments_count or 0,
                        shares_count=post.shares_count,
                        published_at=post.published_date.isoformat() if post.published_date else None,
                        audience=post.audience,
                        post_id=post.post_id,
                        # Merge subtitle/description - keep whichever is longer
                        subtitle=post.subtitle if len(post.subtitle or "") >= len(post.description or "") else post.description,
                        post_type=post.post_type,
                        word_count=post.word_count,
                    ))

                logger.info("source_scraped", handle=handle, posts=len(posts))

            except Exception as e:
                logger.error("source_scrape_failed", handle=handle, error=str(e))
                failed_handles.append(handle)
                continue

            completed += 1
            ctx.report_progress(
                int(completed / total_handles * 100),
                f"Scraped {completed}/{total_handles} publications ({len(all_source_posts)} posts, {len(failed_handles)} failed)"
            )

        # Scrape target publication
        target_posts = []

        if target_handle:
            try:
                metadata, posts = await fetch_publication_posts(target_handle)

                for post in posts:
                    target_posts.append(PostMetadata(
                        url=post.url,
                        title=post.title,
                        author=post.author,
                        publication_handle=target_handle,
                        likes_count=post.likes_count or 0,
                        comments_count=post.comments_count or 0,
                        shares_count=post.shares_count,
                        published_at=post.published_date.isoformat() if post.published_date else None,
                        audience=post.audience,
                        post_id=post.post_id,
                        # Merge subtitle/description - keep whichever is longer
                        subtitle=post.subtitle if len(post.subtitle or "") >= len(post.description or "") else post.description,
                        post_type=post.post_type,
                        word_count=post.word_count,
                    ))

                logger.info("target_scraped", handle=target_handle, posts=len(posts))

            except Exception as e:
                logger.error("target_scrape_failed", handle=target_handle, error=str(e))
                failed_handles.append(target_handle)

            completed += 1
            ctx.report_progress(
                100,
                f"Scraped {completed}/{total_handles} publications ({len(all_source_posts) + len(target_posts)} posts total)"
            )

        logger.info(
            "post_listings_fetched",
            source_count=len(all_source_posts),
            target_count=len(target_posts),
            failed_handles=failed_handles,
        )

        # Show sample posts with all fields so user knows what's available
        def post_to_dict(p):
            return {
                "url": p.url,
                "title": p.title,
                "author": p.author,
                "publication_handle": p.publication_handle,
                "post_id": p.post_id,
                "subtitle": p.subtitle[:100] if p.subtitle else None,
                "post_type": p.post_type,
                "word_count": p.word_count,
                "audience": p.audience,
                "likes_count": p.likes_count,
                "comments_count": p.comments_count,
                "shares_count": p.shares_count,
                "published_at": p.published_at,
            }

        sample_source_posts = [post_to_dict(p) for p in all_source_posts[:5]]
        sample_target_posts = [post_to_dict(p) for p in target_posts[:3]]

        ctx.report_output({
            "source_publications_scraped": len(source_handles) - len([h for h in failed_handles if h in source_handles]),
            "target_scraped": bool(target_posts),
            "source_posts_count": len(all_source_posts),
            "target_posts_count": len(target_posts),
            "failed_handles": failed_handles,
            "sample_source_posts": sample_source_posts,
            "sample_target_posts": sample_target_posts,
            "status": "success",
        })

        return ScrapeIndexOutput(
            source_posts=all_source_posts,
            target_posts=target_posts,
            status="success",
        )

    finally:
        await scraper.close()


async def scrape_full_content(
    ctx,
    params: ScrapeFullContentInput,
) -> ScrapeFullContentOutput:
    """
    Scrape full content for a list of post URLs.

    Supports authenticated scraping for paid subscriber content via cookies.

    For whitelisted publications (paid_subscription_handles):
    - MUST get full content or the node fails
    - Always use authenticated client
    - Force refresh (skip cache for partial content)

    Accepts:
    - posts: List of post objects with url, title, publication_handle (preferred)
    - urls: List of URLs (legacy, will try to extract handle from URL)
    - substack_cookies: Optional dict with session cookies for paid content access
    - paid_subscription_handles: Whitelist - fail if can't get full content from these
    """
    posts = params.posts or []
    urls = params.urls or []
    publication_handle = params.publication_handle
    substack_cookies = params.substack_cookies
    paid_subscription_handles = set(h.lower() for h in (params.paid_subscription_handles or []))

    # Build URL-to-metadata mapping from posts
    post_metadata_by_url = {}
    for post in posts:
        url = post.get("url") if isinstance(post, dict) else post.url
        post_metadata_by_url[url] = post

    # If we have posts, use their URLs; otherwise use urls param
    if posts and not urls:
        urls = [post.get("url") if isinstance(post, dict) else post.url for post in posts]

    # Set up authentication if cookies provided
    auth = None
    authenticated_client = None
    if substack_cookies:
        auth = SubstackAuth(cookies_dict=substack_cookies)
        if auth.is_authenticated:
            # Create authenticated client for paywalled content
            authenticated_client = httpx.AsyncClient(
                timeout=30,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                },
                cookies=auth.cookies,
            )
            logger.info("authenticated_scraping_enabled", cookie_count=len(auth.cookies))

    # Report input
    ctx.report_input({
        "urls_count": len(urls),
        "posts_with_metadata": len(post_metadata_by_url),
        "publication_handle": publication_handle,
        "authenticated": auth.is_authenticated if auth else False,
        "paid_subscription_handles": list(paid_subscription_handles),
    })

    items = []
    failed_urls = []
    cache_hits = 0
    total = len(urls)

    # Report at start
    if total > 0:
        ctx.report_progress(0, f"Starting to scrape {total} posts...")
    else:
        ctx.report_progress(100, "No posts to scrape")

    # Progress frequency: every item if small list, every 5 if large
    progress_every = 1 if total <= 10 else 5

    whitelist_failures = []  # Track failures for whitelisted publications

    try:
        for i, url in enumerate(urls):
            try:
                # Check if this post is paywalled (from metadata)
                post_meta = post_metadata_by_url.get(url, {})
                audience = post_meta.get("audience") if isinstance(post_meta, dict) else getattr(post_meta, "audience", None)
                is_paywalled = audience in ('only_paid', 'founding')

                # Get handle early to check whitelist
                if isinstance(post_meta, dict):
                    handle = post_meta.get("publication_handle") or publication_handle or _extract_handle(url)
                else:
                    handle = getattr(post_meta, "publication_handle", None) or publication_handle or _extract_handle(url)

                is_whitelisted = (handle or "").lower() in paid_subscription_handles

                # Use authenticated client for whitelisted or paywalled posts if available
                client_to_use = None
                force_refresh = False

                if is_whitelisted and authenticated_client:
                    # Whitelisted: always use auth, always force refresh
                    client_to_use = authenticated_client
                    force_refresh = True
                    logger.debug("whitelisted_fetch", url=url, handle=handle)
                elif is_paywalled and authenticated_client:
                    # Paywalled non-whitelisted: use auth, force refresh
                    client_to_use = authenticated_client
                    force_refresh = True
                    logger.debug("authenticated_fetch", url=url, is_paywalled=is_paywalled)

                if client_to_use:
                    html_content, from_cache = await fetch_with_cache(
                        url,
                        cache_ttl_hours=168,  # 7 days for post content
                        client=client_to_use,
                        force_refresh=force_refresh,
                    )
                else:
                    # Use cached fetch for free content
                    html_content, from_cache = await fetch_with_cache(url, cache_ttl_hours=168)

                if from_cache:
                    cache_hits += 1

                soup = BeautifulSoup(html_content, 'lxml')

                # Extract title
                title = None
                title_tag = soup.find('h1', class_='post-title')
                if title_tag:
                    title = title_tag.get_text(strip=True)
                else:
                    og_title = soup.find('meta', property='og:title')
                    if og_title:
                        title = og_title.get('content')

                # Extract content
                content_raw = None
                body_div = soup.find('div', class_='body')
                if body_div:
                    content_raw = body_div.get_text(separator='\n\n', strip=True)
                else:
                    article = soup.find('article')
                    if article:
                        content_raw = article.get_text(separator='\n\n', strip=True)

                # Paywall status already determined above from metadata
                has_full_content = True

                # For paywalled posts, check if we actually got truncated content
                if is_paywalled and content_raw:
                    # Check for HTML paywall indicators (in case we only got partial)
                    paywall_div = soup.find('div', class_='paywall')
                    paywall_truncated = soup.find('div', class_='paywall-truncated')
                    paywall_text = soup.find(string=lambda t: t and any(phrase in t for phrase in [
                        'This post is for paid subscribers',
                        'This post is for paying subscribers',
                        'Subscribe to continue reading',
                        'Already a paid subscriber?',
                    ]))

                    if paywall_div or paywall_truncated or paywall_text:
                        has_full_content = False
                    # Short content on paid post is likely truncated (unless authenticated)
                    elif len(content_raw) < 5000 and not authenticated_client:
                        has_full_content = False

                # CRITICAL: For whitelisted publications, we MUST get full content
                if is_whitelisted and is_paywalled and not has_full_content:
                    logger.error(
                        "whitelist_content_access_failed",
                        url=url,
                        handle=handle,
                        reason="Could not get full content for whitelisted publication - cookies may have expired"
                    )
                    whitelist_failures.append({
                        "url": url,
                        "handle": handle,
                        "title": title or "Unknown",
                    })
                    continue  # Don't add partial content for whitelisted pubs

                # Extract author
                author = None
                author_meta = soup.find('meta', {'name': 'author'})
                if author_meta:
                    author = author_meta.get('content')

                # Get metrics from metadata if available
                if isinstance(post_meta, dict):
                    likes_count = post_meta.get("likes_count", 0)
                    comments_count = post_meta.get("comments_count", 0)
                    published_at = post_meta.get("published_at")
                else:
                    likes_count = getattr(post_meta, "likes_count", 0)
                    comments_count = getattr(post_meta, "comments_count", 0)
                    published_at = getattr(post_meta, "published_at", None)

                items.append(ScrapedItem(
                    url=url,
                    title=title or "Untitled",
                    content_raw=content_raw,
                    author=author,
                    publication_handle=handle,
                    likes_count=likes_count,
                    comments_count=comments_count,
                    published_at=published_at,
                    is_paywalled=is_paywalled,
                    has_full_content=has_full_content,
                ))

                logger.debug(
                    "content_scraped",
                    url=url,
                    title=title,
                    from_cache=from_cache,
                    is_paywalled=is_paywalled,
                    has_full_content=has_full_content,
                    is_whitelisted=is_whitelisted,
                    authenticated=bool(client_to_use),
                )

            except Exception as e:
                logger.error("content_scrape_failed", url=url, error=str(e))
                failed_urls.append(url)

            # Report progress
            if (i + 1) % progress_every == 0 or (i + 1) == total:
                cache_rate = f"{cache_hits}/{i+1} cached" if cache_hits > 0 else "no cache"
                ctx.report_progress(
                    int((i + 1) / total * 100),
                    f"Scraped {i + 1}/{total} posts ({cache_rate}, {len(failed_urls)} failed)"
                )

    finally:
        # Clean up authenticated client
        if authenticated_client:
            await authenticated_client.aclose()

    logger.info(
        "full_content_scrape_complete",
        success=len(items),
        failed=len(failed_urls),
        whitelist_failures=len(whitelist_failures),
        cache_hits=cache_hits,
    )

    # Determine status
    # CRITICAL: Whitelist failures = workflow must fail (cookies expired or misconfiguration)
    if whitelist_failures:
        status = "whitelist_failed"
        logger.error(
            "whitelist_access_failed",
            failures=whitelist_failures,
            message="Could not get full content for whitelisted publications - check cookies"
        )
    elif items:
        status = "success"
    elif not urls:
        status = "success"  # Nothing to do is not a failure
    else:
        status = "all_failed"  # We had URLs but all failed

    # Count paywalled items
    paywalled_count = sum(1 for i in items if i.is_paywalled)
    partial_content_count = sum(1 for i in items if not i.has_full_content)

    # Report output - show actual scraped items with cache and paywall info
    ctx.report_output({
        "count": len(items),
        "cache_hits": cache_hits,
        "fresh_fetches": len(items) - cache_hits,
        "failed": len(failed_urls),
        "whitelist_failures": whitelist_failures,
        "paywalled": paywalled_count,
        "partial_content": partial_content_count,
        "items": [
            {
                "url": i.url,
                "title": i.title,
                "has_content": bool(i.content_raw),
                "is_paywalled": i.is_paywalled,
                "has_full_content": i.has_full_content,
            }
            for i in items
        ],
        "status": status,
    })

    return ScrapeFullContentOutput(
        items=items,
        failed_urls=failed_urls,
        status=status,
    )


async def scrape_metrics_batch(
    ctx,
    params: ScrapeMetricsBatchInput,
) -> ScrapeMetricsBatchOutput:
    """
    Re-scrape engagement metrics for a batch of posts.

    Step 1.4: Update metrics for recent posts (delta update).
    """
    items = params.items

    # Report input
    ctx.report_input({
        "items": items,
    })

    # Group items by publication for efficient API calls
    by_publication = {}
    for item in items:
        handle = item.get("publication_handle")
        if handle:
            if handle not in by_publication:
                by_publication[handle] = []
            by_publication[handle].append(item["url"])

    updates = []

    for handle, urls in by_publication.items():
        try:
            # Fetch all posts from publication (API returns metrics)
            _, posts = await fetch_publication_posts(handle)

            # Create lookup by URL
            post_by_url = {p.url: p for p in posts}

            # Match requested URLs with fetched posts
            for url in urls:
                if url in post_by_url:
                    post = post_by_url[url]
                    updates.append(MetricsUpdate(
                        url=url,
                        likes_count=post.likes_count or 0,
                        comments_count=post.comments_count or 0,
                        shares_count=post.shares_count,
                    ))

            logger.info("metrics_refreshed", handle=handle, count=len(urls))

        except Exception as e:
            logger.error("metrics_refresh_failed", handle=handle, error=str(e))
            continue

    # Report output - show actual updates
    ctx.report_output({
        "updates": [u.model_dump() for u in updates],
        "status": "success",
    })

    return ScrapeMetricsBatchOutput(
        updates=updates,
        count=len(updates),
        status="success",
    )


def _extract_handle(url: str) -> str:
    """
    Extract Substack handle from URL.

    Examples:
        "https://stratechery.substack.com" -> "stratechery"
        "https://stratechery.substack.com/p/some-post" -> "stratechery"
    """
    if not url:
        return None

    # Remove protocol
    url = url.replace("https://", "").replace("http://", "")

    # Extract handle from subdomain
    if ".substack.com" in url:
        return url.split(".substack.com")[0]

    return None


def _normalize_handle(value: str) -> str:
    """
    Normalize a handle - accepts either a handle or URL.

    Examples:
        "stratechery" -> "stratechery"
        "https://stratechery.substack.com" -> "stratechery"
        "  designgurus  " -> "designgurus"
    """
    if not value:
        return ""

    value = value.strip()

    # If it looks like a URL, extract handle
    if "substack.com" in value or "://" in value:
        return _extract_handle(value) or value

    return value


# =============================================================================
# NEW UNIFIED POSTS FUNCTIONS (using SubstackClient)
# =============================================================================

def _substack_post_to_post_data(post: SubstackPost) -> PostData:
    """Convert SubstackPost to PostData schema."""
    return PostData(
        url=post.url,
        publication_handle=post.publication_handle,
        title=post.title,
        subtitle=post.subtitle,
        content_raw=post.content_raw,
        author=post.author,
        substack_post_id=post.substack_post_id,
        slug=post.slug,
        post_type=post.post_type,
        audience=post.audience,
        word_count=post.word_count,
        is_paywalled=post.is_paywalled,
        has_full_content=post.has_full_content,
        likes_count=post.likes_count,
        comments_count=post.comments_count,
        shares_count=post.shares_count,
        published_at=post.published_at.isoformat() if post.published_at else None,
    )


from pydantic import BaseModel, Field


class FetchPostListingsInput(BaseModel):
    """Input for fetch_post_listings node."""
    source_handles: List[str] = Field(default_factory=list, description="Source publication handles")
    substack_cookies: Optional[Dict[str, str]] = Field(default=None, description="Substack cookies for auth")


class FetchPostListingsOutput(BaseModel):
    """Output from fetch_post_listings node."""
    posts: List[PostData] = Field(default_factory=list)
    handles_scraped: int = 0
    handles_failed: int = 0
    failed_handles: List[str] = Field(default_factory=list)
    status: str = "success"


async def fetch_post_listings(
    ctx,
    params: FetchPostListingsInput,
) -> FetchPostListingsOutput:
    """
    Fetch post listings from source publications using SubstackClient.

    This is the new unified version that returns PostData for the posts table.
    Posts are global - no target association.

    Input:
    - source_handles: List of Substack publication handles
    - substack_cookies: Optional cookies for authenticated access

    Output:
    - posts: List of PostData objects (metadata only, no content)
    - handles_scraped: Number of handles successfully scraped
    - handles_failed: Number of handles that failed
    """
    source_handles = params.source_handles
    # Get cookies from params or ctx.secrets (project vault)
    substack_cookies = params.substack_cookies
    if not substack_cookies:
        # Build cookies dict from individual secrets
        sid = ctx.secrets.get("SUBSTACK_SID")
        lli = ctx.secrets.get("SUBSTACK_LLI")
        if sid and lli:
            substack_cookies = {
                "substack.sid": sid,
                "substack.lli": lli,
            }

    # Normalize handles
    source_handles = [_normalize_handle(h) for h in source_handles if h]

    ctx.report_input({
        "source_handles": source_handles,
        "has_cookies": bool(substack_cookies),
        "cookies_source": "params" if params.substack_cookies else ("secrets" if substack_cookies else "none"),
    })

    if not source_handles:
        ctx.report_output({
            "posts_count": 0,
            "status": "success",
            "message": "No handles provided",
        })
        return FetchPostListingsOutput(
            posts=[],
            handles_scraped=0,
            handles_failed=0,
            status="success",
        )

    # Create client with optional auth
    # Caching is controlled per-method in SubstackClient:
    # - fetch_all_posts: Cached 1 hour (listings change, but not every minute)
    # - fetch_post_content: Cached 7 days (content rarely changes)
    client = await create_client(cookies=substack_cookies)

    all_posts = []
    failed_handles = []
    handles_scraped = 0
    total_handles = len(source_handles)
    total_cache_hits = 0
    total_pages_fetched = 0

    # Delay between handles to avoid 429 rate limiting
    HANDLE_DELAY = 2.0  # 2 seconds between different publications
    import asyncio

    try:
        for i, handle in enumerate(source_handles):
            try:
                posts, cache_hits, pages_fetched = await client.fetch_all_posts(
                    handle, return_cache_status=True
                )
                total_cache_hits += cache_hits
                total_pages_fetched += pages_fetched

                for post in posts:
                    all_posts.append(_substack_post_to_post_data(post))

                handles_scraped += 1
                logger.info("handle_scraped", handle=handle, posts=len(posts), cache_hits=cache_hits, pages=pages_fetched)

            except Exception as e:
                logger.error("handle_scrape_failed", handle=handle, error=str(e))
                failed_handles.append(handle)

            # Report progress with cache info
            cache_info = f"{total_cache_hits}/{total_pages_fetched} cached" if total_pages_fetched > 0 else ""
            ctx.report_progress(
                int((i + 1) / total_handles * 100),
                f"Scraped {i + 1}/{total_handles} handles ({len(all_posts)} posts, {cache_info})"
            )

            # Delay between handles to avoid rate limiting
            if i < total_handles - 1:  # Don't delay after last handle
                await asyncio.sleep(HANDLE_DELAY)

        logger.info(
            "post_listings_fetched",
            total_posts=len(all_posts),
            handles_scraped=handles_scraped,
            handles_failed=len(failed_handles),
        )

        # Fail if any handle failed to fetch - don't continue with partial data
        if failed_handles:
            ctx.report_output({
                "posts_count": len(all_posts),
                "handles_scraped": handles_scraped,
                "handles_failed": len(failed_handles),
                "failed_handles": failed_handles,
                "cache_hits": total_cache_hits,
                "total_pages": total_pages_fetched,
                "fresh_fetches": total_pages_fetched - total_cache_hits,
                "status": "error",
                "error": f"Failed to fetch {len(failed_handles)} handle(s): {', '.join(failed_handles)}",
            })
            return FetchPostListingsOutput(
                posts=all_posts,
                handles_scraped=handles_scraped,
                handles_failed=len(failed_handles),
                failed_handles=failed_handles,
                status="error",
            )

        ctx.report_output({
            "posts_count": len(all_posts),
            "handles_scraped": handles_scraped,
            "handles_failed": 0,
            "failed_handles": [],
            "cache_hits": total_cache_hits,
            "total_pages": total_pages_fetched,
            "fresh_fetches": total_pages_fetched - total_cache_hits,
            "sample_posts": [
                {"url": p.url, "title": p.title[:50] if p.title else "", "likes": p.likes_count}
                for p in all_posts[:5]
            ],
            "status": "success",
        })

        return FetchPostListingsOutput(
            posts=all_posts,
            handles_scraped=handles_scraped,
            handles_failed=0,
            failed_handles=[],
            status="success",
        )

    finally:
        await client.close()


class FetchPostContentBatchInput(BaseModel):
    """Input for fetch_post_content_batch node."""
    posts: List[PostData] = Field(default_factory=list, description="Posts to fetch content for")
    substack_cookies: Optional[Dict[str, str]] = Field(default=None, description="Substack cookies for auth")
    paid_subscription_handles: List[str] = Field(default_factory=list, description="Whitelist of paid subscriptions")


class FetchPostContentBatchOutput(BaseModel):
    """Output from fetch_post_content_batch node."""
    posts: List[PostData] = Field(default_factory=list)
    fetched_count: int = 0
    failed_count: int = 0
    failed_urls: List[str] = Field(default_factory=list)
    status: str = "success"


async def fetch_post_content_batch(
    ctx,
    params: FetchPostContentBatchInput,
) -> FetchPostContentBatchOutput:
    """
    Fetch full content for posts and store each one immediately to the database.

    This is a "fetch-and-store" approach - each post is saved as soon as it's
    fetched, so no work is lost if the workflow fails partway through.

    Failure handling:
    - Uses 1.5s delay between requests to avoid rate limiting
    - Tolerates up to 10 consecutive failures before aborting
    - At the end, fails if ANY posts couldn't be fetched (so next run retries them)

    Input:
    - posts: List of PostData objects (with metadata)
    - substack_cookies: Optional cookies for authenticated access
    - paid_subscription_handles: Whitelist - fail if can't get full content from these

    Output:
    - posts: List of PostData that were stored (for downstream nodes if needed)
    - fetched_count: Number successfully fetched and stored
    - failed_count: Number that failed to fetch
    - stored_count: Number successfully stored to database
    """
    from datetime import datetime, timezone
    from shared.database import get_db_session
    from shared.models import Post
    from sqlalchemy import select, and_
    import asyncio

    # Configuration
    NORMAL_DELAY = 1.0  # Seconds between successful requests (matches REQUEST_DELAY in client)
    MAX_CONSECUTIVE_FAILURES = 10  # Abort after this many failures in a row
    # Exponential backoff: 5s, 10s, 20s, 30s (capped)
    BACKOFF_DELAYS = [5, 10, 20, 30]

    posts = params.posts
    # Get cookies from params or ctx.secrets (project vault)
    substack_cookies = params.substack_cookies
    if not substack_cookies:
        # Build cookies dict from individual secrets
        sid = ctx.secrets.get("SUBSTACK_SID")
        lli = ctx.secrets.get("SUBSTACK_LLI")
        if sid and lli:
            substack_cookies = {
                "substack.sid": sid,
                "substack.lli": lli,
            }

    # Get paid subscription handles from params or secrets
    paid_handles_param = params.paid_subscription_handles or ctx.secrets.get("PAID_SUBSCRIPTION_HANDLES") or []
    # Handle different formats: array, comma-separated string
    if isinstance(paid_handles_param, str):
        paid_handles_param = [h.strip() for h in paid_handles_param.split(",") if h.strip()]
    elif isinstance(paid_handles_param, list):
        paid_handles_param = [str(h).strip() for h in paid_handles_param if h]
    paid_subscription_handles = set(h.lower() for h in paid_handles_param)

    ctx.report_input({
        "posts_count": len(posts),
        "has_cookies": bool(substack_cookies),
        "paid_subscription_handles": list(paid_subscription_handles),
        "cookies_source": "params" if params.substack_cookies else ("secrets" if substack_cookies else "none"),
    })

    if not posts:
        ctx.report_progress(100, "No posts to fetch")
        ctx.report_output({
            "fetched_count": 0,
            "stored_count": 0,
            "status": "success",
            "message": "No posts to fetch",
        })
        return FetchPostContentBatchOutput(
            posts=[],
            fetched_count=0,
            failed_count=0,
            status="success",
        )

    # On retry attempts, filter out posts that were already stored in previous attempts
    # This prevents re-fetching the same 33 posts on each retry
    already_stored_count = 0
    if ctx.attempt > 1:
        async with get_db_session() as db:
            # Get URLs of posts that are already stored with content
            post_urls = [p.url for p in posts]
            result = await db.execute(
                select(Post.url).where(
                    and_(
                        Post.url.in_(post_urls),
                        Post.content_raw.isnot(None),
                    )
                )
            )
            stored_urls = set(row[0] for row in result.fetchall())

            # Filter to only posts not yet stored
            original_count = len(posts)
            posts = [p for p in posts if p.url not in stored_urls]
            already_stored_count = original_count - len(posts)

            if already_stored_count > 0:
                logger.info(
                    "retry_filtered_already_stored",
                    attempt=ctx.attempt,
                    original_count=original_count,
                    already_stored=already_stored_count,
                    remaining=len(posts),
                )

        if not posts:
            ctx.report_progress(100, f"All {already_stored_count} posts already stored in previous attempt")
            ctx.report_output({
                "fetched_count": 0,
                "stored_count": already_stored_count,
                "status": "success",
                "message": f"All posts already stored in previous attempt",
            })
            return FetchPostContentBatchOutput(
                posts=[],
                fetched_count=0,
                failed_count=0,
                status="success",
            )

    # Create client with optional auth
    client = await create_client(cookies=substack_cookies)

    # Counters
    fetched_count = 0
    stored_count = 0
    failed_urls = []
    result_posts = []
    total = len(posts)
    consecutive_failures = 0

    # Report initial progress (with different message on retry)
    if ctx.attempt == 1:
        ctx.report_progress(0, f"Fetching and storing {total} posts...")
    else:
        ctx.report_progress(0, f"Retry attempt {ctx.attempt}: {total} remaining posts ({already_stored_count} already stored)")

    try:
        for i, post_data in enumerate(posts):
            handle = post_data.publication_handle.lower()
            is_whitelisted = handle in paid_subscription_handles
            fetch_success = False

            from_cache = False  # Default for error cases
            # Use the new status-based API for clean error handling
            result = await client.fetch_post_content_with_status(
                post_data.url,
                force_refresh=is_whitelisted,
            )
            from_cache = result.from_cache

            # Handle non-success cases explicitly
            if result.is_not_found:
                # Post no longer exists - skip it, don't count as failure
                logger.warning("post_not_found_skipping", url=post_data.url)
                # Don't add to failed_urls, don't increment consecutive_failures
                # Progress will still update, but this post is simply skipped
            elif not result.is_success:
                # Other errors (rate limited, auth required, etc.)
                if result.is_retriable:
                    logger.error("post_fetch_failed", url=post_data.url, status=result.status, error=result.error)
                    failed_urls.append(post_data.url)
                    consecutive_failures += 1
                else:
                    # Non-retriable errors (e.g., auth required) - log but don't retry
                    logger.warning("post_fetch_skipped", url=post_data.url, status=result.status, error=result.error)
            elif result.post:
                fetched = result.post
                # Merge metadata from original post
                fetched.publication_handle = post_data.publication_handle
                fetched.substack_post_id = post_data.substack_post_id
                fetched.slug = post_data.slug
                fetched.subtitle = post_data.subtitle or fetched.subtitle
                fetched.post_type = post_data.post_type
                fetched.audience = post_data.audience
                fetched.word_count = post_data.word_count
                fetched.likes_count = post_data.likes_count
                fetched.comments_count = post_data.comments_count
                fetched.shares_count = post_data.shares_count

                # Parse published_at
                if post_data.published_at:
                    try:
                        fetched.published_at = datetime.fromisoformat(
                            post_data.published_at.replace("Z", "+00:00")
                        )
                    except (ValueError, AttributeError):
                        pass

                # Update paywall status from metadata
                if post_data.audience in ('only_paid', 'founding'):
                    fetched.is_paywalled = True

                # For whitelisted publications, check if we got full content
                is_founding_tier = post_data.audience == 'founding'
                if is_whitelisted and fetched.is_paywalled and not fetched.has_full_content:
                    if not is_founding_tier:
                        # Regular paid post we should have access to
                        logger.warning(
                            "whitelist_partial_content",
                            url=post_data.url,
                            handle=handle,
                        )

                # Store to database immediately
                async with get_db_session() as db:
                    db_result = await db.execute(
                        select(Post).where(Post.url == fetched.url)
                    )
                    existing = db_result.scalar_one_or_none()

                    if existing:
                        # Update existing post
                        existing.title = fetched.title
                        existing.publication_handle = fetched.publication_handle
                        if fetched.subtitle:
                            existing.subtitle = fetched.subtitle
                        if fetched.author:
                            existing.author = fetched.author
                        if fetched.substack_post_id:
                            existing.substack_post_id = fetched.substack_post_id
                        if fetched.slug:
                            existing.slug = fetched.slug
                        if fetched.post_type:
                            existing.post_type = fetched.post_type
                        if fetched.audience:
                            existing.audience = fetched.audience
                        if fetched.word_count:
                            existing.word_count = fetched.word_count
                        existing.is_paywalled = fetched.is_paywalled
                        if fetched.likes_count:
                            existing.likes_count = fetched.likes_count
                        if fetched.comments_count:
                            existing.comments_count = fetched.comments_count
                        if fetched.shares_count:
                            existing.shares_count = fetched.shares_count
                        if fetched.published_at:
                            existing.published_at = fetched.published_at

                        # Only update content if new is better
                        if fetched.content_raw:
                            should_update = (
                                not existing.content_raw or
                                fetched.has_full_content or
                                not existing.has_full_content
                            )
                            if should_update:
                                existing.content_raw = fetched.content_raw
                                existing.content_fetched_at = datetime.now(timezone.utc)
                                existing.has_full_content = fetched.has_full_content

                        existing.metrics_updated_at = datetime.now(timezone.utc)
                    else:
                        # Insert new post
                        new_post = Post(
                            url=fetched.url,
                            publication_handle=fetched.publication_handle,
                            title=fetched.title,
                            subtitle=fetched.subtitle,
                            content_raw=fetched.content_raw,
                            author=fetched.author,
                            substack_post_id=fetched.substack_post_id,
                            slug=fetched.slug,
                            post_type=fetched.post_type,
                            audience=fetched.audience,
                            word_count=fetched.word_count,
                            is_paywalled=fetched.is_paywalled,
                            has_full_content=fetched.has_full_content,
                            likes_count=fetched.likes_count or 0,
                            comments_count=fetched.comments_count or 0,
                            shares_count=fetched.shares_count or 0,
                            published_at=fetched.published_at,
                            content_fetched_at=datetime.now(timezone.utc),
                            metrics_updated_at=datetime.now(timezone.utc),
                            discovery_source="content_sync",
                        )
                        db.add(new_post)

                    # Commit is automatic via get_db_session context manager

                fetched_count += 1
                stored_count += 1
                result_posts.append(_substack_post_to_post_data(fetched))
                fetch_success = True
                consecutive_failures = 0  # Reset on success
                logger.debug("post_fetched_and_stored", url=post_data.url)

            # Check if we should abort due to too many consecutive failures
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error(
                    "aborting_too_many_consecutive_failures",
                    consecutive_failures=consecutive_failures,
                    stored_so_far=stored_count,
                )
                ctx.report_output({
                    "fetched_count": fetched_count,
                    "stored_count": stored_count,
                    "failed_count": len(failed_urls),
                    "failed_urls": failed_urls[:20],
                    "status": "error",
                    "error": f"Aborted after {consecutive_failures} consecutive failures. Stored {stored_count} posts before aborting.",
                })
                return FetchPostContentBatchOutput(
                    posts=result_posts,
                    fetched_count=fetched_count,
                    failed_count=len(failed_urls),
                    failed_urls=failed_urls,
                    status="error",
                )

            # Smart delay based on result:
            # - No delay if content came from cache
            # - Normal delay (0.3s) on success
            # - Exponential backoff on failure: 2s, 4s, 8s, 16s (capped)
            if fetch_success and from_cache:
                pass  # No delay for cached content
            elif fetch_success:
                await asyncio.sleep(NORMAL_DELAY)
            else:
                backoff_idx = min(consecutive_failures - 1, len(BACKOFF_DELAYS) - 1)
                backoff_delay = BACKOFF_DELAYS[backoff_idx]
                logger.debug("backoff_delay", consecutive_failures=consecutive_failures, delay=backoff_delay)
                await asyncio.sleep(backoff_delay)

            # Report progress
            completed = i + 1
            percent = int(completed / total * 100)
            ctx.report_progress(
                percent,
                f"Processed {completed}/{total} ({stored_count} stored, {len(failed_urls)} failed)"
            )

        # Determine final status
        # Fail if ANY posts couldn't be fetched - next run will retry them
        if failed_urls:
            status = "error"
            error_msg = f"Failed to fetch {len(failed_urls)}/{total} posts. Successfully stored {stored_count}."
        else:
            status = "success"
            error_msg = None

        logger.info(
            "fetch_and_store_complete",
            total=total,
            fetched=fetched_count,
            stored=stored_count,
            failed=len(failed_urls),
            status=status,
        )

        ctx.report_output({
            "fetched_count": fetched_count,
            "stored_count": stored_count,
            "failed_count": len(failed_urls),
            "failed_urls": failed_urls[:20],
            "status": status,
            "error": error_msg,
        })

        return FetchPostContentBatchOutput(
            posts=result_posts,
            fetched_count=fetched_count,
            failed_count=len(failed_urls),
            failed_urls=failed_urls,
            status=status,
        )

    finally:
        await client.close()
