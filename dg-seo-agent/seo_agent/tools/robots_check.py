"""robots.txt compliance checker — verifies URLs are allowed before scraping."""

import logging
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

logger = logging.getLogger(__name__)

_parsers: dict[str, RobotFileParser | None] = {}

USER_AGENT = "SEOBot"


def _get_base_url(url: str) -> str:
    """Extract scheme + netloc from a URL (e.g. https://example.com)."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _get_robot_parser(base_url: str) -> RobotFileParser | None:
    """Fetch and parse robots.txt for a domain. Returns None if unreachable."""
    if base_url in _parsers:
        return _parsers[base_url]

    robots_url = f"{base_url}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)

    try:
        response = httpx.get(robots_url, timeout=10, follow_redirects=True)
        if response.status_code == 200:
            rp.parse(response.text.splitlines())
            _parsers[base_url] = rp
            logger.debug("Loaded robots.txt from %s", robots_url)
            return rp
        else:
            # No robots.txt or non-200 = assume everything is allowed
            logger.debug("No robots.txt at %s (status %d) — allowing all", robots_url, response.status_code)
            _parsers[base_url] = None
            return None
    except Exception as e:
        logger.warning("Failed to fetch robots.txt from %s: %s", robots_url, e)
        _parsers[base_url] = None
        return None


def can_scrape(url: str, user_agent: str = USER_AGENT) -> bool:
    """Check if the given URL is allowed by robots.txt.

    Returns True if:
    - robots.txt allows the URL for our user agent
    - robots.txt doesn't exist or is unreachable (permissive default)
    """
    base_url = _get_base_url(url)
    rp = _get_robot_parser(base_url)

    if rp is None:
        # No robots.txt = allowed
        return True

    allowed = rp.can_fetch(user_agent, url)
    if not allowed:
        logger.info("robots.txt DISALLOWS scraping: %s", url)
    return allowed


def get_crawl_delay(url: str, user_agent: str = USER_AGENT) -> float | None:
    """Get the crawl-delay directive for our user agent, if any."""
    base_url = _get_base_url(url)
    rp = _get_robot_parser(base_url)

    if rp is None:
        return None

    try:
        delay = rp.crawl_delay(user_agent)
        return float(delay) if delay is not None else None
    except Exception:
        return None


def clear_robots_cache() -> None:
    """Clear the cached robots.txt parsers (useful for testing)."""
    _parsers.clear()
