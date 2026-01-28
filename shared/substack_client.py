"""
Unified Substack client for all Substack API interactions.

This client consolidates all Substack fetching functionality:
- Post index fetching (metadata + metrics) - NO CACHING (data changes frequently)
- Full content fetching - CACHED (content rarely changes)
- Publication metadata
- Integrates with SubstackAuth for paid content access

Caching strategy:
- Post listings (fetch_post_index, fetch_all_posts): Never cached - new posts appear frequently
- Post content (fetch_post_content): Cached for 7 days - content rarely changes
"""
import httpx
import json
import asyncio
import structlog
from bs4 import BeautifulSoup
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from urllib.parse import urlparse

from .substack_scraper import SubstackAuth

logger = structlog.get_logger()

# Cache TTL for post content (7 days in hours)
CONTENT_CACHE_TTL_HOURS = 168

# Rate limiting
REQUEST_DELAY = 0.2  # 200ms between requests


@dataclass
class SubstackPost:
    """Represents a Substack post with all metadata."""
    url: str
    title: str
    publication_handle: str

    # Optional identification
    substack_post_id: Optional[str] = None
    slug: Optional[str] = None

    # Author and dates
    author: Optional[str] = None
    published_at: Optional[datetime] = None

    # Content (only populated by fetch_post_content)
    subtitle: Optional[str] = None
    description: Optional[str] = None
    content_raw: Optional[str] = None

    # Substack metadata
    post_type: Optional[str] = None  # 'newsletter', 'podcast', 'thread'
    audience: Optional[str] = None   # 'everyone', 'only_paid', 'founding'
    word_count: Optional[int] = None

    # Tags and categorization
    tags: List[str] = field(default_factory=list)  # Post tags (e.g., "System Design", "Coding Interviews")
    section_name: Optional[str] = None  # Section within publication (e.g., "Deep Dives", "News Roundup")

    # Engagement metrics
    likes_count: int = 0
    comments_count: int = 0
    shares_count: int = 0

    # Paywall tracking
    is_paywalled: bool = False
    has_full_content: bool = True


class SubstackClient:
    """
    Unified client for all Substack API interactions.

    Caching strategy is built into each method:
    - Post listings: No cache (new posts appear frequently)
    - Post content: Cached for 7 days (content rarely changes)
    """

    def __init__(
        self,
        auth: Optional[SubstackAuth] = None,
        timeout: int = 30,
    ):
        """
        Initialize the client.

        Args:
            auth: Optional SubstackAuth for accessing paid content
            timeout: Request timeout in seconds
        """
        self.auth = auth
        self.timeout = timeout

        # Build headers
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        # Build cookies dict for httpx
        self._cookies = None
        if auth and auth.is_authenticated:
            self._cookies = auth.cookies
            logger.info("substack_client_authenticated", cookie_count=len(self._cookies))

        self._client: Optional[httpx.AsyncClient] = None

    @property
    def is_authenticated(self) -> bool:
        """Check if client has valid authentication."""
        return self.auth is not None and self.auth.is_authenticated

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers=self._headers,
                cookies=self._cookies,
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _fetch(self, url: str) -> str:
        """
        Simple HTTP fetch with no caching.

        Use for data that changes frequently (post listings, APIs).
        """
        client = await self._get_client()
        response = await client.get(url)
        response.raise_for_status()
        logger.debug("fetched", url=url)
        return response.text

    async def _fetch_with_cache(
        self,
        url: str,
        cache_ttl_hours: int = CONTENT_CACHE_TTL_HOURS,
        force_refresh: bool = False,
    ) -> Tuple[str, bool]:
        """
        HTTP fetch with caching.

        Use for content that rarely changes (individual post content).

        Returns:
            Tuple of (content, from_cache)
        """
        from nodes.http_cache import fetch_with_cache
        client = await self._get_client()
        return await fetch_with_cache(
            url=url,
            cache_ttl_hours=cache_ttl_hours,
            client=client,
            force_refresh=force_refresh,
        )

    async def fetch_post_index(
        self,
        handle: str,
        sort: str = "new",
        limit: int = 12,
    ) -> List[SubstackPost]:
        """
        Fetch post listings (metadata only) from a publication.

        Cached for 1 hour - balances freshness with efficiency.

        Args:
            handle: Substack publication handle
            sort: Sort order - 'new' or 'top'
            limit: Maximum posts to fetch

        Returns:
            List of SubstackPost with metadata (no content)
        """
        base_url = f"https://{handle}.substack.com"
        api_url = f"{base_url}/api/v1/archive?sort={sort}&limit={limit}"

        try:
            content, _ = await self._fetch_with_cache(api_url, cache_ttl_hours=1)
            data = json.loads(content)

            posts = []
            for post_data in data[:limit]:
                post = self._parse_post_metadata(post_data, handle, base_url)
                if post:
                    posts.append(post)

            logger.info("post_index_fetched", handle=handle, sort=sort, count=len(posts))
            return posts

        except Exception as e:
            logger.error("post_index_fetch_failed", handle=handle, error=str(e))
            raise

    async def fetch_all_posts(
        self,
        handle: str,
        max_pages: int = 20,
        page_size: int = 12,
    ) -> List[SubstackPost]:
        """
        Fetch all posts from a publication (paginated).

        Cached for 1 hour per page - balances freshness with efficiency.

        Args:
            handle: Substack publication handle
            max_pages: Maximum pages to fetch (safety limit)
            page_size: Posts per page

        Returns:
            List of all SubstackPost with metadata
        """
        base_url = f"https://{handle}.substack.com"
        posts = []
        offset = 0

        try:
            for page in range(max_pages):
                api_url = f"{base_url}/api/v1/archive?sort=new&offset={offset}&limit={page_size}"

                content, _ = await self._fetch_with_cache(api_url, cache_ttl_hours=1)
                data = json.loads(content)

                if not data:
                    break  # No more posts

                for post_data in data:
                    post = self._parse_post_metadata(post_data, handle, base_url)
                    if post:
                        posts.append(post)

                if len(data) < page_size:
                    break  # Last page

                offset += page_size
                await asyncio.sleep(REQUEST_DELAY)

            logger.info("all_posts_fetched", handle=handle, count=len(posts))
            return posts

        except Exception as e:
            logger.error("all_posts_fetch_failed", handle=handle, error=str(e))
            raise

    async def fetch_post_content(
        self,
        url: str,
        force_refresh: bool = False,
    ) -> Optional[SubstackPost]:
        """
        Fetch full content for a single post.

        NOTE: Cached for 7 days - post content rarely changes.

        Args:
            url: Full post URL
            force_refresh: Skip cache and fetch fresh

        Returns:
            SubstackPost with content_raw populated, or None on failure
        """
        try:
            # Cache post content for 7 days (168 hours) - content rarely changes
            content, from_cache = await self._fetch_with_cache(
                url,
                cache_ttl_hours=CONTENT_CACHE_TTL_HOURS,
                force_refresh=force_refresh,
            )
            soup = BeautifulSoup(content, 'lxml')

            # Extract handle from URL
            handle = self._extract_handle(url)

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

            # Extract author
            author = None
            author_meta = soup.find('meta', {'name': 'author'})
            if author_meta:
                author = author_meta.get('content')

            # Check for paywall indicators
            is_paywalled = False
            has_full_content = True

            paywall_div = soup.find('div', class_='paywall')
            paywall_truncated = soup.find('div', class_='paywall-truncated')
            paywall_text = soup.find(string=lambda t: t and any(phrase in t for phrase in [
                'This post is for paid subscribers',
                'This post is for paying subscribers',
                'Subscribe to continue reading',
                'Already a paid subscriber?',
            ]))

            if paywall_div or paywall_truncated or paywall_text:
                is_paywalled = True
                has_full_content = False
            elif content_raw and len(content_raw) < 5000 and not self.is_authenticated:
                # Short content might be truncated
                # Check for paywall-specific patterns in HTML
                if 'paywall' in content.lower() or 'paid subscriber' in content.lower():
                    is_paywalled = True
                    has_full_content = False

            post = SubstackPost(
                url=url,
                title=title or "Untitled",
                publication_handle=handle or "",
                author=author,
                content_raw=content_raw,
                is_paywalled=is_paywalled,
                has_full_content=has_full_content,
            )

            logger.debug(
                "post_content_fetched",
                url=url,
                title=title,
                from_cache=from_cache,
                is_paywalled=is_paywalled,
                has_full_content=has_full_content,
            )

            return post

        except Exception as e:
            logger.error("post_content_fetch_failed", url=url, error=str(e))
            return None

    async def fetch_posts_content_batch(
        self,
        posts: List[SubstackPost],
        force_refresh: bool = False,
        whitelisted_handles: Optional[set] = None,
        on_progress: Optional[callable] = None,
    ) -> Tuple[List[SubstackPost], List[str]]:
        """
        Fetch full content for multiple posts.

        Args:
            posts: List of SubstackPost objects (with metadata)
            force_refresh: Skip cache for whitelisted publications
            whitelisted_handles: Set of handles where we must get full content
            on_progress: Optional callback(completed, total, message) for progress reporting

        Returns:
            Tuple of (posts_with_content, failed_urls)
        """
        whitelisted_handles = whitelisted_handles or set()
        results = []
        failed_urls = []
        total = len(posts)

        for i, post in enumerate(posts):
            handle = post.publication_handle.lower()
            is_whitelisted = handle in whitelisted_handles
            should_force = force_refresh or is_whitelisted

            try:
                fetched = await self.fetch_post_content(post.url, force_refresh=should_force)
                if fetched:
                    # Merge metadata from original post with fetched content
                    fetched.publication_handle = post.publication_handle
                    fetched.substack_post_id = post.substack_post_id
                    fetched.slug = post.slug
                    fetched.subtitle = post.subtitle or fetched.subtitle
                    fetched.description = post.description or fetched.description
                    fetched.post_type = post.post_type
                    fetched.audience = post.audience
                    fetched.word_count = post.word_count
                    fetched.likes_count = post.likes_count
                    fetched.comments_count = post.comments_count
                    fetched.shares_count = post.shares_count
                    fetched.published_at = post.published_at

                    # Update paywall status from metadata if not detected in content
                    if post.audience in ('only_paid', 'founding'):
                        fetched.is_paywalled = True
                        # Only mark as partial if we didn't get full content
                        if not fetched.has_full_content:
                            fetched.has_full_content = False

                    # For whitelisted publications, check if we got full content
                    # Skip 'founding' tier posts (e.g., Executive Circle) - higher tier we may not have
                    is_founding_tier = post.audience == 'founding'
                    if is_whitelisted and fetched.is_paywalled and not fetched.has_full_content:
                        if is_founding_tier:
                            # Log but don't fail - we don't have this subscription tier
                            logger.info(
                                "skipping_founding_tier_post",
                                url=post.url,
                                handle=handle,
                                audience=post.audience,
                            )
                            # Still include with partial content
                        else:
                            # Regular paid post we should have access to - this is a real failure
                            logger.error(
                                "whitelist_content_access_failed",
                                url=post.url,
                                handle=handle,
                                audience=post.audience,
                            )
                            failed_urls.append(post.url)
                            continue

                    results.append(fetched)
                else:
                    failed_urls.append(post.url)

            except Exception as e:
                logger.error("post_content_batch_fetch_failed", url=post.url, error=str(e))
                failed_urls.append(post.url)

            await asyncio.sleep(REQUEST_DELAY)

            # Report progress
            if on_progress:
                completed = i + 1
                on_progress(completed, total, f"Fetched {completed}/{total} posts ({len(failed_urls)} failed)")

        return results, failed_urls

    def _parse_post_metadata(
        self,
        post_data: Dict[str, Any],
        handle: str,
        base_url: str,
    ) -> Optional[SubstackPost]:
        """Parse post metadata from Substack API response."""
        try:
            slug = post_data.get('slug', '')
            if not slug:
                logger.warning("post_missing_slug", handle=handle, title=post_data.get("title"))
                return None

            # Parse published date
            post_date = post_data.get('post_date')
            published_at = None
            if post_date:
                if isinstance(post_date, str):
                    published_at = datetime.fromisoformat(post_date.replace('Z', '+00:00'))

            # Extract author
            author = 'Unknown'
            bylines = post_data.get('publishedBylines', [])
            if bylines and isinstance(bylines, list) and len(bylines) > 0:
                author = bylines[0].get('name', 'Unknown')

            # Get audience type
            audience = post_data.get('audience')
            is_paywalled = audience in ('only_paid', 'founding')

            # Get Substack's numeric ID
            substack_post_id = post_data.get('id')
            if substack_post_id:
                substack_post_id = str(substack_post_id)

            # Extract tags from postTags
            tags = []
            for tag in post_data.get('postTags', []):
                if tag.get('name') and not tag.get('hidden'):
                    tags.append(tag['name'])

            # Extract section name
            section_name = post_data.get('section_name') or None

            return SubstackPost(
                url=f"{base_url}/p/{slug}",
                title=post_data.get('title', 'Untitled'),
                publication_handle=handle,
                substack_post_id=substack_post_id,
                slug=slug,
                author=author,
                published_at=published_at,
                subtitle=post_data.get('subtitle'),
                description=post_data.get('description'),
                post_type=post_data.get('type'),
                audience=audience,
                word_count=post_data.get('wordcount'),
                tags=tags,
                section_name=section_name,
                likes_count=post_data.get('reaction_count', 0) or 0,
                comments_count=post_data.get('comment_count', 0) or 0,
                shares_count=post_data.get('share_count', 0) or 0,
                is_paywalled=is_paywalled,
                has_full_content=True,  # Will be updated when content is fetched
            )

        except Exception as e:
            logger.warning("post_metadata_parse_failed", slug=post_data.get('slug'), error=str(e))
            return None

    def _extract_handle(self, url: str) -> Optional[str]:
        """Extract Substack handle from URL."""
        if not url:
            return None

        # Remove protocol
        url = url.replace("https://", "").replace("http://", "")

        # Extract handle from subdomain
        if ".substack.com" in url:
            return url.split(".substack.com")[0]

        return None


# Convenience function
async def create_client(
    cookies: Optional[Dict[str, str]] = None,
    # Legacy params - ignored, kept for backwards compatibility
    use_cache: bool = True,
    cache_days: int = 7,
) -> SubstackClient:
    """
    Create a SubstackClient with optional authentication.

    Args:
        cookies: Substack session cookies (substack.sid, substack.lli)
        use_cache: DEPRECATED - caching is now decided per-method
        cache_days: DEPRECATED - caching is now decided per-method

    Returns:
        Configured SubstackClient
    """
    auth = None
    if cookies:
        auth = SubstackAuth(cookies_dict=cookies)

    return SubstackClient(auth=auth)
