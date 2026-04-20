"""Competitor scraper — fetches and analyses competitor pages."""

import asyncio
import logging
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from seo_agent.cache.manager import async_cached_call
from seo_agent.state import CompetitorData
from seo_agent.tools.robots_check import can_scrape

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
SCRAPE_SEMAPHORE = asyncio.Semaphore(3)
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds


def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def get_meta_description(soup: BeautifulSoup) -> str:
    """Extract meta description from parsed HTML."""
    tag = soup.find("meta", attrs={"name": "description"})
    if tag and tag.get("content"):
        return tag["content"].strip()
    # Also check og:description
    tag = soup.find("meta", attrs={"property": "og:description"})
    if tag and tag.get("content"):
        return tag["content"].strip()
    return ""


def _parse_page(url: str, html: str) -> CompetitorData:
    """Parse HTML content into structured competitor data."""
    soup = BeautifulSoup(html, "html.parser")
    domain = extract_domain(url)

    # Count links
    internal_links = 0
    external_links = 0
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Skip anchors and javascript
        if href.startswith("#") or href.startswith("javascript:"):
            continue
        link_domain = extract_domain(href)
        if link_domain == "" or link_domain == domain:
            internal_links += 1
        else:
            external_links += 1

    return {
        "url": url,
        "title": soup.find("title").get_text().strip() if soup.find("title") else "",
        "meta_description": get_meta_description(soup),
        "h1": [h.get_text().strip() for h in soup.find_all("h1")],
        "h2": [h.get_text().strip() for h in soup.find_all("h2")],
        "h3": [h.get_text().strip() for h in soup.find_all("h3")],
        "word_count": len(soup.get_text().split()),
        "has_schema": bool(soup.find("script", {"type": "application/ld+json"})),
        "internal_links": internal_links,
        "external_links": external_links,
    }


async def _fetch_page(url: str) -> str:
    """Fetch a page with retry logic and exponential backoff."""
    headers = {"User-Agent": USER_AGENT}

    for attempt in range(MAX_RETRIES):
        try:
            async with SCRAPE_SEMAPHORE:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        url, headers=headers, timeout=15, follow_redirects=True
                    )
                    response.raise_for_status()
                    return response.text
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "Fetch attempt %d/%d failed for %s: %s — retrying in %ds",
                    attempt + 1, MAX_RETRIES, url, e, delay,
                )
                await asyncio.sleep(delay)
            else:
                raise


async def scrape_page(url: str) -> CompetitorData:
    """Scrape a single page: fetch HTML and parse into structured data.

    Checks robots.txt before fetching. Uses caching and retry with backoff.
    """
    if not can_scrape(url):
        logger.info("Skipping %s — blocked by robots.txt", url)
        return {
            "url": url,
            "title": "[blocked by robots.txt]",
            "meta_description": "",
            "h1": [], "h2": [], "h3": [],
            "word_count": 0,
            "has_schema": False,
            "internal_links": 0,
            "external_links": 0,
        }

    html = await async_cached_call(_fetch_page, url)
    return _parse_page(url, html)


async def scrape_competitors(urls: list[str]) -> list[CompetitorData]:
    """Scrape multiple competitor URLs concurrently (max 3 at a time via semaphore)."""
    tasks = [scrape_page(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    scraped = []
    for url, result in zip(urls, results):
        if isinstance(result, Exception):
            logger.error("Failed to scrape %s: %s", url, result)
            scraped.append({
                "url": url,
                "title": f"[error: {result}]",
                "meta_description": "",
                "h1": [], "h2": [], "h3": [],
                "word_count": 0,
                "has_schema": False,
                "internal_links": 0,
                "external_links": 0,
            })
        else:
            scraped.append(result)

    return scraped


def analyse_competitors_node(state: dict) -> dict:
    """LangGraph node: scrape and analyse competitor pages for all keywords."""
    results = state.get("results", [])
    errors = state.get("errors", [])

    for kw_data in results:
        keyword = kw_data["keyword"]
        competitor_urls = [c["url"] for c in kw_data.get("top_competitors", [])]

        if not competitor_urls:
            logger.info("No competitors to scrape for keyword '%s'", keyword)
            continue

        try:
            scraped = asyncio.run(scrape_competitors(competitor_urls))
            kw_data["raw_competitor_data"] = scraped
            logger.info(
                "Scraped %d competitors for keyword '%s'",
                len(scraped), keyword,
            )
        except Exception as e:
            error_msg = f"[analyse_competitors] keyword='{keyword}' error: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    return {**state, "results": results, "errors": errors}
