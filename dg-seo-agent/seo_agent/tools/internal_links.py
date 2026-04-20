"""Internal link auditor — analyses internal linking structure for a target page."""

import asyncio
import logging
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from seo_agent.cache.manager import async_cached_call
from seo_agent.tools.competitor import USER_AGENT

logger = logging.getLogger(__name__)

CRAWL_SEMAPHORE = asyncio.Semaphore(3)


def _get_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _is_internal(href: str, base_url: str) -> bool:
    """Check if a link is internal (same domain or relative)."""
    if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
        return False
    if href.startswith("/") or href.startswith("./") or href.startswith("../"):
        return True
    return _get_domain(href) == _get_domain(base_url)


def _normalize_url(href: str, base_url: str) -> str:
    """Resolve a potentially relative URL against a base URL."""
    return urljoin(base_url, href).split("#")[0].split("?")[0].rstrip("/")


async def _fetch_page(url: str) -> str:
    """Fetch a page's HTML."""
    async with CRAWL_SEMAPHORE:
        headers = {"User-Agent": USER_AGENT}
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=15, follow_redirects=True)
            response.raise_for_status()
            return response.text


def _extract_internal_links(html: str, page_url: str) -> list[dict]:
    """Extract all internal links from a page with their anchor text."""
    soup = BeautifulSoup(html, "html.parser")
    domain = _get_domain(page_url)
    links = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if _is_internal(href, page_url):
            resolved = _normalize_url(href, page_url)
            anchor = a.get_text().strip()
            if resolved and resolved != _normalize_url(page_url, page_url):
                links.append({
                    "url": resolved,
                    "anchor_text": anchor,
                })

    return links


def _calculate_link_score(
    outgoing_internal: int,
    incoming_internal: int,
    relevant_anchors: int,
    total_anchors: int,
) -> float:
    """Calculate internal link score (0.0 - 1.0).

    Factors:
    - Outgoing internal links from the page (target: 5-20)
    - Incoming internal links to the page from other pages (target: 3+)
    - Anchor text relevance (% of anchors containing keyword terms)
    """
    # Outgoing score: 0-20 mapped to 0-1, capped at 1
    outgoing_score = min(outgoing_internal / 10, 1.0)

    # Incoming score: 0-5 mapped to 0-1, capped at 1
    incoming_score = min(incoming_internal / 5, 1.0)

    # Anchor relevance score
    anchor_score = (relevant_anchors / total_anchors) if total_anchors > 0 else 0.0

    # Weighted average
    score = (outgoing_score * 0.3) + (incoming_score * 0.5) + (anchor_score * 0.2)
    return round(min(score, 1.0), 2)


def _check_anchor_relevance(anchor_text: str, keyword: str) -> bool:
    """Check if anchor text is relevant to the keyword."""
    if not anchor_text or not keyword:
        return False
    anchor_lower = anchor_text.lower()
    keyword_terms = keyword.lower().split()
    # At least one keyword term appears in anchor
    return any(term in anchor_lower for term in keyword_terms)


async def audit_internal_links(
    target_url: str,
    keyword: str,
    site_base_url: str,
    sample_pages: int = 10,
) -> dict:
    """Audit internal linking for a target page.

    Checks:
    1. How many internal links the target page has (outgoing)
    2. Samples other pages on the site to find incoming links to target
    3. Evaluates anchor text relevance
    4. Identifies specific issues

    Args:
        target_url: The page URL to audit.
        keyword: The target keyword (for anchor relevance check).
        site_base_url: The site's base URL (e.g. https://example.com).
        sample_pages: Number of site pages to sample for incoming link check.

    Returns:
        Dict with: score, issues, outgoing_count, incoming_count
    """
    issues: list[str] = []

    # Step 1: Fetch target page and extract outgoing internal links
    try:
        target_html = await async_cached_call(_fetch_page, target_url)
    except Exception as e:
        logger.error("Failed to fetch target page %s: %s", target_url, e)
        return {
            "score": 0.0,
            "issues": [f"Could not fetch target page: {e}"],
            "outgoing_count": 0,
            "incoming_count": 0,
        }

    outgoing_links = _extract_internal_links(target_html, target_url)
    outgoing_count = len(outgoing_links)

    if outgoing_count == 0:
        issues.append("Page has zero internal links — add contextual links to related pages")
    elif outgoing_count < 3:
        issues.append(f"Page has only {outgoing_count} internal links — aim for 5-20 contextual links")
    elif outgoing_count > 100:
        issues.append(f"Page has {outgoing_count} internal links — too many, focus on most relevant")

    # Check for generic anchor text
    generic_anchors = ["click here", "read more", "learn more", "here", "link", "this"]
    generic_count = sum(
        1 for link in outgoing_links
        if link["anchor_text"].lower().strip() in generic_anchors
    )
    if generic_count > 0:
        issues.append(
            f"{generic_count} internal links use generic anchor text "
            "(e.g. 'click here') — use descriptive, keyword-rich anchors"
        )

    # Step 2: Fetch homepage to find site navigation links to the target
    incoming_count = 0
    incoming_anchors: list[str] = []

    try:
        homepage_html = await async_cached_call(_fetch_page, site_base_url)
        homepage_links = _extract_internal_links(homepage_html, site_base_url)
        target_normalized = _normalize_url(target_url, site_base_url)

        for link in homepage_links:
            if _normalize_url(link["url"], site_base_url) == target_normalized:
                incoming_count += 1
                incoming_anchors.append(link["anchor_text"])

        # Also check a few pages linked from homepage for deeper incoming links
        sample_urls = list({link["url"] for link in homepage_links})[:sample_pages]

        async def check_page_for_link(page_url: str) -> list[str]:
            try:
                html = await async_cached_call(_fetch_page, page_url)
                links = _extract_internal_links(html, page_url)
                anchors = []
                for link in links:
                    if _normalize_url(link["url"], site_base_url) == target_normalized:
                        anchors.append(link["anchor_text"])
                return anchors
            except Exception:
                return []

        tasks = [check_page_for_link(url) for url in sample_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                incoming_count += len(result)
                incoming_anchors.extend(result)

    except Exception as e:
        logger.warning("Could not check incoming links from homepage: %s", e)
        issues.append("Could not analyse incoming internal links (homepage fetch failed)")

    if incoming_count == 0:
        issues.append(
            "No internal links pointing to this page found — "
            "this may be an orphan page, add links from related content"
        )
    elif incoming_count < 3:
        issues.append(
            f"Only {incoming_count} internal links point to this page — "
            "add more contextual links from blog posts and related pages"
        )

    # Step 3: Check anchor text relevance
    all_anchors = incoming_anchors
    relevant_anchors = sum(
        1 for a in all_anchors if _check_anchor_relevance(a, keyword)
    )
    total_anchors = len(all_anchors)

    if total_anchors > 0 and relevant_anchors == 0:
        issues.append(
            f"None of the {total_anchors} incoming anchor texts contain keyword terms "
            f"('{keyword}') — update anchor text to be more descriptive"
        )

    # Calculate score
    score = _calculate_link_score(
        outgoing_count, incoming_count, relevant_anchors, total_anchors
    )

    return {
        "score": score,
        "issues": issues,
        "outgoing_count": outgoing_count,
        "incoming_count": incoming_count,
    }


def audit_internal_links_node(state: dict) -> dict:
    """LangGraph node: audit internal links for each keyword's target URL."""
    results = state.get("results", [])
    errors = state.get("errors", [])
    target_domain = state["target_domain"]

    # Normalize base URL
    if not target_domain.startswith("http"):
        site_base = f"https://{target_domain}"
    else:
        site_base = target_domain.rstrip("/")

    for kw_data in results:
        keyword = kw_data["keyword"]
        url = kw_data.get("your_url")

        if not url:
            url = site_base

        try:
            audit = asyncio.run(audit_internal_links(url, keyword, site_base))
            kw_data["internal_link_score"] = audit["score"]
            kw_data["internal_link_issues"] = audit["issues"]
            logger.info(
                "Internal links for '%s': score=%.2f, %d issues, "
                "%d outgoing, %d incoming",
                keyword, audit["score"], len(audit["issues"]),
                audit["outgoing_count"], audit["incoming_count"],
            )
        except Exception as e:
            error_msg = f"[audit_internal_links] keyword='{keyword}' error: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    return {**state, "results": results, "errors": errors}
