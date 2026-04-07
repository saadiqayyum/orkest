"""
Substack publication scraper.
Fetches posts from Substack publications using their API with RSS fallback.

Supports authenticated scraping for paid subscriber content via cookies.
"""
import httpx
import feedparser
import json
import os
from bs4 import BeautifulSoup
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
from urllib.parse import urlparse
from pathlib import Path
import re
import structlog

logger = structlog.get_logger()


@dataclass
class SubstackAuth:
    """
    Authentication for Substack using browser session cookies.

    Supports two ways to provide cookies:
    1. cookies_path: Path to JSON file with exported cookies
    2. cookies_dict: Dictionary with cookie values directly

    Required cookies:
    - substack.sid: Session ID
    - substack.lli: Authentication token

    To export cookies from browser:
    1. Login to Substack in your browser
    2. Open DevTools (F12) > Application > Cookies
    3. Copy 'substack.sid' and 'substack.lli' values
    """
    cookies_path: Optional[str] = None
    cookies_dict: Optional[Dict[str, str]] = None
    _cookies: Dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if self.cookies_path:
            self._load_from_file(self.cookies_path)
        elif self.cookies_dict:
            self._cookies = self.cookies_dict

    def _load_from_file(self, path: str):
        """Load cookies from JSON file (browser export format)."""
        try:
            with open(path) as f:
                data = json.load(f)

            # Handle different export formats
            if isinstance(data, list):
                # Browser extension format: [{name, value, domain}, ...]
                for cookie in data:
                    if cookie.get('domain', '').endswith('substack.com'):
                        self._cookies[cookie['name']] = cookie['value']
            elif isinstance(data, dict):
                # Simple format: {name: value, ...}
                self._cookies = data

            logger.info("substack_cookies_loaded",
                       count=len(self._cookies),
                       keys=list(self._cookies.keys()))
        except Exception as e:
            logger.error("substack_cookies_load_failed", path=path, error=str(e))
            raise

    @property
    def cookies(self) -> Dict[str, str]:
        return self._cookies

    @property
    def is_authenticated(self) -> bool:
        """Check if we have the required cookies for authentication."""
        return 'substack.sid' in self._cookies or 'substack.lli' in self._cookies

    def get_cookie_header(self) -> str:
        """Get cookies as a header string."""
        return '; '.join(f'{k}={v}' for k, v in self._cookies.items())


@dataclass
class SubstackPost:
    """Represents a Substack post from Substack API."""
    post_id: str
    title: str
    author: str
    published_date: datetime
    likes_count: int = 0
    shares_count: int = 0
    comments_count: int = 0
    url: str = ""
    # Audience: 'everyone' (free), 'only_paid', 'founding', or None (unknown)
    audience: Optional[str] = None
    # Additional fields from Substack API
    subtitle: Optional[str] = None
    description: Optional[str] = None
    post_type: Optional[str] = None  # 'newsletter', 'podcast', 'thread'
    word_count: Optional[int] = None


@dataclass
class PublicationMetadata:
    """Substack publication metadata."""
    handle: str
    name: Optional[str] = None
    author: Optional[str] = None
    url: str = ""
    total_posts: int = 0


class SubstackScraper:
    """
    Scraper for Substack publications.

    Supports optional authentication for accessing paid subscriber content.
    """

    def __init__(self, auth: Optional[SubstackAuth] = None, timeout: int = 30):
        """
        Initialize the scraper.

        Args:
            auth: Optional SubstackAuth for accessing paid content
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.auth = auth

        # Build headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        # Build cookies dict for httpx
        cookies = None
        if auth and auth.is_authenticated:
            cookies = auth.cookies
            logger.info("substack_scraper_authenticated", cookie_count=len(cookies))

        self.client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers=headers,
            cookies=cookies,
        )

    @property
    def is_authenticated(self) -> bool:
        """Check if scraper has valid authentication."""
        return self.auth is not None and self.auth.is_authenticated

    async def fetch_publication_metadata(self, handle: str) -> PublicationMetadata:
        """
        Fetch publication metadata.

        Args:
            handle: Substack publication handle

        Returns:
            Publication metadata
        """
        url = f"https://{handle}.substack.com"

        try:
            response = await self.client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'lxml')

            # Try to extract publication name
            name = None
            name_meta = soup.find('meta', property='og:site_name')
            if name_meta:
                name = name_meta.get('content')
            else:
                title_tag = soup.find('title')
                if title_tag:
                    name = title_tag.text.strip()

            # Try to extract author
            author = None
            author_meta = soup.find('meta', {'name': 'author'})
            if author_meta:
                author = author_meta.get('content')

            logger.info("publication_metadata_fetched", handle=handle, name=name)

            return PublicationMetadata(
                handle=handle,
                name=name,
                author=author,
                url=url,
                total_posts=0
            )

        except Exception as e:
            logger.error("publication_metadata_fetch_failed", handle=handle, error=str(e))
            raise

    async def fetch_all_posts(self, handle: str) -> List[SubstackPost]:
        """
        Fetch all posts from a Substack publication using Substack's API.

        Args:
            handle: Substack publication handle

        Returns:
            List of posts with metadata including engagement metrics
        """
        try:
            posts = await self._fetch_from_api(handle)
            logger.info("posts_fetched", handle=handle, count=len(posts))
            return posts

        except Exception as e:
            logger.error("post_fetch_failed", handle=handle, error=str(e))
            logger.info("falling_back_to_rss", handle=handle)
            return await self._fetch_from_rss(handle)

    async def _fetch_from_api(self, handle: str) -> List[SubstackPost]:
        """
        Fetch posts from Substack's internal API.

        This API provides engagement metrics (likes, comments, shares).
        Uses /api/v1/archive endpoint with pagination.
        """
        base_url = f"https://{handle}.substack.com"
        posts = []
        offset = 0
        page_size = 12
        max_pages = 20  # Safety limit: 240 posts max

        try:
            for page in range(max_pages):
                api_url = f"{base_url}/api/v1/archive?sort=new&offset={offset}&limit={page_size}"

                logger.debug("fetching_api_page", url=api_url, page=page)

                response = await self.client.get(api_url)
                response.raise_for_status()

                data = response.json()

                # Empty response means we've fetched all posts
                if not data:
                    break

                for post_data in data:
                    try:
                        # Extract post ID from slug
                        slug = post_data.get('slug', '')
                        post_id = f"p-{slug}" if slug else self._extract_post_id(post_data.get('canonical_url', ''))

                        # Parse published date
                        post_date = post_data.get('post_date')
                        if post_date:
                            if isinstance(post_date, str):
                                published_date = datetime.fromisoformat(post_date.replace('Z', '+00:00'))
                            else:
                                published_date = datetime.now()
                        else:
                            published_date = datetime.now()

                        # Get engagement metrics
                        likes_count = post_data.get('reaction_count', 0) or 0
                        comments_count = post_data.get('comment_count', 0) or 0
                        shares_count = post_data.get('share_count', 0) or 0

                        # Build URL
                        post_url = post_data.get('canonical_url') or f"{base_url}/p/{slug}"

                        # Extract author
                        author = 'Unknown'
                        bylines = post_data.get('publishedBylines', [])
                        if bylines and isinstance(bylines, list) and len(bylines) > 0:
                            author = bylines[0].get('name', 'Unknown')

                        # Get audience type (free vs paid)
                        # 'everyone' = free, 'only_paid' = paid subscribers, 'founding' = founding members
                        audience = post_data.get('audience')

                        # Get additional metadata
                        subtitle = post_data.get('subtitle')
                        description = post_data.get('description')
                        post_type = post_data.get('type')  # 'newsletter', 'podcast', 'thread'
                        word_count = post_data.get('wordcount')

                        # Use Substack's numeric ID if available
                        numeric_id = post_data.get('id')
                        if numeric_id:
                            post_id = str(numeric_id)

                        post = SubstackPost(
                            post_id=post_id,
                            title=post_data.get('title', 'Untitled'),
                            author=author,
                            published_date=published_date,
                            likes_count=likes_count,
                            comments_count=comments_count,
                            shares_count=shares_count,
                            url=post_url,
                            audience=audience,
                            subtitle=subtitle,
                            description=description,
                            post_type=post_type,
                            word_count=word_count,
                        )
                        posts.append(post)

                    except Exception as e:
                        logger.warning("post_parse_failed", slug=post_data.get('slug'), error=str(e))
                        continue

                # Check if we got fewer posts than requested (last page)
                if len(data) < page_size:
                    break

                offset += page_size

            logger.info("api_posts_fetched", handle=handle, count=len(posts))
            return posts

        except Exception as e:
            logger.error("api_fetch_failed", handle=handle, error=str(e))
            raise

    async def _fetch_from_rss(self, handle: str) -> List[SubstackPost]:
        """
        Fetch posts from RSS feed.

        Note: RSS feed provides basic metadata but not engagement metrics.
        """
        rss_url = f"https://{handle}.substack.com/feed"

        try:
            response = await self.client.get(rss_url)
            response.raise_for_status()

            feed = feedparser.parse(response.text)
            posts = []

            for entry in feed.entries:
                try:
                    post_url = entry.link
                    post_id = self._extract_post_id(post_url)
                    published_date = datetime(*entry.published_parsed[:6])
                    author = entry.get('author', 'Unknown')

                    post = SubstackPost(
                        post_id=post_id,
                        title=entry.title,
                        author=author,
                        published_date=published_date,
                        likes_count=0,
                        shares_count=0,
                        comments_count=0,
                        url=post_url
                    )
                    posts.append(post)

                except Exception as e:
                    logger.warning("post_parse_failed", title=entry.get('title'), error=str(e))
                    continue

            logger.debug("rss_posts_parsed", count=len(posts))
            return posts

        except Exception as e:
            logger.error("rss_fetch_failed", handle=handle, error=str(e))
            raise

    def _extract_post_id(self, url: str) -> str:
        """Extract post ID from Substack URL."""
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')

        if 'p' in path_parts:
            idx = path_parts.index('p')
            if idx + 1 < len(path_parts):
                return f"p-{path_parts[idx + 1]}"

        return parsed.path.replace('/', '-').strip('-')

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# Convenience functions
async def fetch_publication_posts(
    handle: str,
    auth: Optional[SubstackAuth] = None,
) -> tuple[PublicationMetadata, List[SubstackPost]]:
    """
    Fetch publication metadata and all posts.

    Args:
        handle: Substack publication handle (e.g., 'postsyntax')
        auth: Optional authentication for paid content access

    Returns:
        Tuple of (metadata, posts)
    """
    scraper = SubstackScraper(auth=auth)
    try:
        metadata = await scraper.fetch_publication_metadata(handle)
        posts = await scraper.fetch_all_posts(handle)
        metadata.total_posts = len(posts)
        return metadata, posts
    finally:
        await scraper.close()


async def fetch_post_content(
    url: str,
    auth: Optional[SubstackAuth] = None,
) -> str:
    """
    Fetch full HTML content for a single post.

    Args:
        url: Full post URL
        auth: Optional authentication for paid content access

    Returns:
        HTML content of the post
    """
    scraper = SubstackScraper(auth=auth)
    try:
        response = await scraper.client.get(url)
        response.raise_for_status()
        return response.text
    finally:
        await scraper.close()
