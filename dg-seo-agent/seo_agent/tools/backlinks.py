"""Backlink gap analysis — finds domains linking to competitors but not you via Moz API."""

import asyncio
import logging
from collections import Counter

import requests
from requests.auth import HTTPBasicAuth

from seo_agent.cache.manager import cached_call
from seo_agent.config import settings

logger = logging.getLogger(__name__)

MOZ_API_BASE = "https://lsapi.seomoz.com/v2"


def _moz_request(endpoint: str, body: dict, moz_id: str, moz_secret: str) -> dict:
    """Make an authenticated request to the Moz API."""
    url = f"{MOZ_API_BASE}/{endpoint}"
    response = requests.post(
        url,
        json=body,
        auth=HTTPBasicAuth(moz_id, moz_secret),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _get_linking_domains(target: str) -> list[dict]:
    """Fetch linking (referring) domains for a target URL/domain via Moz.

    Returns list of dicts: [{domain, domain_authority}]
    """
    body = {
        "target": target,
        "scope": "root_domain",
        "limit": 50,
    }
    data = cached_call(
        _moz_request,
        "linking_root_domains",
        body,
        settings.moz_access_id,
        settings.moz_secret_key,
    )

    results = []
    for item in data.get("results", []):
        results.append({
            "domain": item.get("root_domain", ""),
            "domain_authority": item.get("domain_authority", 0),
        })

    return results


def _get_domain_authority(target: str) -> float:
    """Get the Domain Authority for a target via Moz URL Metrics."""
    body = {
        "targets": [target],
    }
    data = cached_call(
        _moz_request,
        "url_metrics",
        body,
        settings.moz_access_id,
        settings.moz_secret_key,
    )

    results = data.get("results", [])
    if results:
        return results[0].get("domain_authority", 0)
    return 0


def find_backlink_gaps(
    target_domain: str,
    competitor_domains: list[str],
    max_competitors: int = 3,
) -> list[dict]:
    """Find domains linking to competitors but not to you.

    Args:
        target_domain: Your domain (e.g. 'example.com')
        competitor_domains: List of competitor domains
        max_competitors: Max competitors to analyse (Moz free tier is limited)

    Returns:
        List of dicts sorted by domain_authority (highest first):
        [{domain, domain_authority, linked_competitors}]
    """
    # Limit competitors to avoid burning API quota
    competitors_to_check = competitor_domains[:max_competitors]

    # Get your linking domains
    logger.info("Fetching backlink profile for %s", target_domain)
    your_links = _get_linking_domains(target_domain)
    your_domains = {item["domain"].lower() for item in your_links}

    # Get competitor linking domains
    competitor_link_sets: list[set[str]] = []
    all_competitor_links: dict[str, float] = {}  # domain -> best DA

    for comp_domain in competitors_to_check:
        try:
            logger.info("Fetching backlink profile for competitor %s", comp_domain)
            comp_links = _get_linking_domains(comp_domain)
            comp_domain_set = set()
            for item in comp_links:
                d = item["domain"].lower()
                comp_domain_set.add(d)
                # Track the highest DA we've seen for this domain
                if d not in all_competitor_links or item["domain_authority"] > all_competitor_links[d]:
                    all_competitor_links[d] = item["domain_authority"]
            competitor_link_sets.append(comp_domain_set)
        except Exception as e:
            logger.error("Failed to fetch backlinks for %s: %s", comp_domain, e)
            competitor_link_sets.append(set())

    # Find domains linking to 2+ competitors but NOT linking to you
    domain_counts = Counter()
    for comp_set in competitor_link_sets:
        for domain in comp_set:
            domain_counts[domain] += 1

    gaps = []
    for domain, count in domain_counts.items():
        if count >= 2 and domain not in your_domains:
            gaps.append({
                "domain": domain,
                "domain_authority": all_competitor_links.get(domain, 0),
                "linked_competitors": count,
            })

    # Sort by domain authority (highest first)
    gaps.sort(key=lambda x: x["domain_authority"], reverse=True)

    logger.info(
        "Found %d backlink gap domains for %s (checked %d competitors)",
        len(gaps), target_domain, len(competitors_to_check),
    )

    return gaps


def find_backlink_gaps_node(state: dict) -> dict:
    """LangGraph node: find backlink gaps for each keyword."""
    results = state.get("results", [])
    errors = state.get("errors", [])

    if not settings.moz_access_id or not settings.moz_secret_key:
        logger.warning("MOZ_ACCESS_ID or MOZ_SECRET_KEY not set — skipping backlink analysis")
        for kw_data in results:
            kw_data["backlink_gap"] = []
        return {**state, "results": results, "errors": errors}

    target_domain = state["target_domain"]
    # Normalize: strip protocol for Moz
    if "://" in target_domain:
        from urllib.parse import urlparse
        target_domain = urlparse(target_domain).netloc

    for kw_data in results:
        keyword = kw_data["keyword"]
        competitor_domains = [c["domain"] for c in kw_data.get("top_competitors", [])]

        if not competitor_domains:
            logger.info("No competitors for backlink analysis on '%s'", keyword)
            continue

        try:
            gaps = find_backlink_gaps(target_domain, competitor_domains)
            kw_data["backlink_gap"] = [
                f"{g['domain']} (DA: {g['domain_authority']:.0f}, links to {g['linked_competitors']} competitors)"
                for g in gaps[:20]  # Top 20 opportunities
            ]
            logger.info(
                "Backlink gaps for '%s': %d opportunities found",
                keyword, len(kw_data["backlink_gap"]),
            )
        except Exception as e:
            error_msg = f"[find_backlink_gaps] keyword='{keyword}' error: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    return {**state, "results": results, "errors": errors}
