"""On-page SEO auditor — checks a target page against SEO best practices.

Also persists the parsed target-page data as `your_page_data` so the
competitor_insights node can compare structural metrics directly.
"""

import asyncio
import logging

import httpx
from bs4 import BeautifulSoup

from seo_agent.cache.manager import async_cached_call
from seo_agent.state import CompetitorData
from seo_agent.tools.competitor import USER_AGENT, _parse_page

logger = logging.getLogger(__name__)


async def _fetch_page(url: str) -> str:
    """Fetch a page's HTML content."""
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers, timeout=15, follow_redirects=True)
        response.raise_for_status()
        return response.text


def _audit_page(url: str, html: str, keyword: str, avg_competitor_word_count: int) -> list[str]:
    """Run all on-page SEO checks and return a list of issues found."""
    soup = BeautifulSoup(html, "html.parser")
    issues: list[str] = []
    keyword_lower = keyword.lower()

    # Title tag
    title_tag = soup.find("title")
    if not title_tag:
        issues.append("Missing title tag")
    else:
        title_text = title_tag.get_text().strip()
        title_len = len(title_text)
        if title_len == 0:
            issues.append("Title tag is empty")
        elif title_len < 50:
            issues.append(f"Title tag is {title_len} characters — too short (aim for 50-60)")
        elif title_len > 60:
            issues.append(f"Title tag is {title_len} characters — too long (aim for 50-60)")

        if title_text and keyword_lower not in title_text.lower():
            issues.append(f"Keyword '{keyword}' not found in title tag")

    # Meta description
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if not meta_desc or not meta_desc.get("content"):
        issues.append("Missing meta description")
    else:
        desc_text = meta_desc["content"].strip()
        desc_len = len(desc_text)
        if desc_len < 150:
            issues.append(f"Meta description is {desc_len} characters — too short (aim for 150-160)")
        elif desc_len > 160:
            issues.append(f"Meta description is {desc_len} characters — too long (aim for 150-160)")

    # H1 tag
    h1_tags = soup.find_all("h1")
    if len(h1_tags) == 0:
        issues.append("No H1 tag found")
    elif len(h1_tags) > 1:
        issues.append(f"Multiple H1 tags found ({len(h1_tags)}) — should be exactly one")

    if h1_tags:
        h1_text = " ".join(h.get_text().strip() for h in h1_tags).lower()
        if keyword_lower not in h1_text:
            issues.append(f"Keyword '{keyword}' not found in H1 tag")

    # Word count vs competitors
    body_text = soup.get_text()
    word_count = len(body_text.split())

    if avg_competitor_word_count > 0:
        threshold = int(avg_competitor_word_count * 0.8)
        if word_count < threshold:
            gap = avg_competitor_word_count - word_count
            issues.append(
                f"Word count ({word_count}) is below 80% of competitor average "
                f"({avg_competitor_word_count}) — add ~{gap} more words"
            )

    # Schema markup
    schema_tags = soup.find_all("script", {"type": "application/ld+json"})
    if not schema_tags:
        issues.append("No schema markup (JSON-LD) found — add structured data")

    # Image alt tags
    images = soup.find_all("img")
    if images:
        missing_alt = [img for img in images if not img.get("alt")]
        if missing_alt:
            issues.append(
                f"{len(missing_alt)} of {len(images)} images missing alt text"
            )

    # Canonical tag
    canonical = soup.find("link", attrs={"rel": "canonical"})
    if not canonical or not canonical.get("href"):
        issues.append("No canonical tag found — add <link rel='canonical'>")

    # H2 headings
    h2_tags = soup.find_all("h2")
    if len(h2_tags) == 0:
        issues.append("No H2 headings found — add subheadings to structure content")

    return issues


def _get_avg_competitor_word_count(raw_competitor_data: list[dict]) -> int:
    """Calculate average word count from competitor data."""
    counts = [c["word_count"] for c in raw_competitor_data if c.get("word_count", 0) > 0]
    if not counts:
        return 0
    return sum(counts) // len(counts)


def audit_onpage_node(state: dict) -> dict:
    """LangGraph node: audit on-page SEO for each keyword's target URL.

    Also persists parsed page data as `your_page_data` for the
    competitor_insights node to compare against competitors.
    """
    results = state.get("results", [])
    errors = state.get("errors", [])
    target_domain = state["target_domain"]

    for kw_data in results:
        keyword = kw_data["keyword"]
        url = kw_data.get("your_url")

        # If not ranking, audit the homepage
        if not url:
            url = target_domain.rstrip("/")
            if not url.startswith("http"):
                url = f"https://{url}"

        try:
            html = asyncio.run(async_cached_call(_fetch_page, url))
            avg_wc = _get_avg_competitor_word_count(kw_data.get("raw_competitor_data", []))
            issues = _audit_page(url, html, keyword, avg_wc)
            kw_data["on_page_issues"] = issues

            # Persist parsed page data (same shape as competitors) for comparison
            kw_data["your_page_data"] = _parse_page(url, html)

            logger.info(
                "On-page audit for '%s' (%s): %d issues found",
                keyword, url, len(issues),
            )
        except Exception as e:
            error_msg = f"[audit_onpage] keyword='{keyword}' url='{url}' error: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            kw_data.setdefault("your_page_data", None)

    return {**state, "results": results, "errors": errors}
