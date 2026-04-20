"""SERP rank checker — fetches rankings, SERP features, and People Also Ask via SerpAPI."""

import logging
from urllib.parse import urlparse

import requests

from seo_agent.cache.manager import cached_call
from seo_agent.config import settings
from seo_agent.state import CompetitorRanking, KeywordData, SERPFeatures

logger = logging.getLogger(__name__)


def extract_domain(url: str) -> str:
    """Extract the domain from a URL (e.g. 'https://example.com/page' -> 'example.com')."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return url


def _fetch_serp(keyword: str, api_key: str) -> dict:
    """Raw SerpAPI call — this is the function we cache."""
    params = {
        "q": keyword,
        "num": 100,
        "api_key": api_key,
        "engine": "google",
    }
    response = requests.get("https://serpapi.com/search", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _extract_serp_features(data: dict) -> SERPFeatures:
    """Extract SERP features from SerpAPI response."""
    return {
        "has_featured_snippet": "answer_box" in data or "featured_snippet" in data,
        "has_knowledge_panel": "knowledge_graph" in data,
        "has_video_carousel": "inline_videos" in data,
        "has_image_pack": "inline_images" in data,
        "has_local_pack": "local_results" in data,
        "people_also_ask": [
            q.get("question", "") for q in data.get("related_questions", [])
        ],
    }


def check_serp_ranking(
    keyword: str,
    target_domain: str,
    max_competitors: int | None = None,
) -> dict:
    """Check SERP ranking for a keyword and extract competitor + feature data.

    Args:
        keyword: The search keyword to check.
        target_domain: Your domain (e.g. 'example.com' or 'https://example.com').
        max_competitors: Max number of competitors to collect. Defaults to settings.

    Returns:
        Dict with: keyword, your_rank, your_url, your_page, top_competitors, serp_features
    """
    if max_competitors is None:
        max_competitors = settings.max_competitors

    # Normalize target domain
    target_domain_clean = extract_domain(target_domain) if "://" in target_domain else target_domain.lower()

    # Fetch SERP data (cached)
    data = cached_call(_fetch_serp, keyword, settings.serpapi_key)

    results = data.get("organic_results", [])

    your_position = None
    your_url = None
    competitors: list[CompetitorRanking] = []

    for i, result in enumerate(results):
        url = result.get("link", "")
        result_domain = extract_domain(url)

        if target_domain_clean in result_domain and your_position is None:
            your_position = i + 1
            your_url = url
        elif len(competitors) < max_competitors:
            competitors.append({
                "rank": i + 1,
                "url": url,
                "title": result.get("title", ""),
                "domain": result_domain,
            })

    # Calculate page number (10 results per page)
    your_page = ((your_position - 1) // 10 + 1) if your_position else None

    # Extract SERP features
    serp_features = _extract_serp_features(data)

    logger.info(
        "Keyword '%s': rank=%s, page=%s, competitors=%d, PAA=%d",
        keyword,
        your_position or "not found",
        your_page or "N/A",
        len(competitors),
        len(serp_features["people_also_ask"]),
    )

    return {
        "keyword": keyword,
        "your_rank": your_position,
        "your_url": your_url,
        "your_page": your_page,
        "top_competitors": competitors,
        "serp_features": serp_features,
    }


def check_rankings_node(state: dict) -> dict:
    """LangGraph node: check SERP rankings for all keywords.

    Processes each keyword independently — errors on one keyword
    don't stop the others.
    """
    target_domain = state["target_domain"]
    keywords = state["keywords"]
    results: list[dict] = state.get("results", [])
    errors: list[str] = state.get("errors", [])

    for keyword in keywords:
        try:
            ranking_data = check_serp_ranking(keyword, target_domain)

            # Initialize a full KeywordData entry with defaults for later nodes
            kw_result: KeywordData = {
                "keyword": keyword,
                "your_url": ranking_data["your_url"],
                "your_rank": ranking_data["your_rank"],
                "your_page": ranking_data["your_page"],
                "top_competitors": ranking_data["top_competitors"],
                "serp_features": ranking_data["serp_features"],
                "on_page_issues": [],
                "missing_topics": [],
                "backlink_gap": [],
                "internal_link_score": 0.0,
                "internal_link_issues": [],
                "page_speed": None,
                "raw_competitor_data": [],
                "your_page_data": None,
                "competitor_insights": [],
            }
            results.append(kw_result)

        except Exception as e:
            error_msg = f"[check_rankings] keyword='{keyword}' error: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    return {**state, "results": results, "errors": errors}
