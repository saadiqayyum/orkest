"""Google PageSpeed Insights — fetches Core Web Vitals for a URL."""

import logging

import requests

from seo_agent.cache.manager import cached_call
from seo_agent.config import settings

logger = logging.getLogger(__name__)

PAGESPEED_API_URL = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"


def _fetch_pagespeed(url: str, api_key: str, strategy: str = "mobile") -> dict:
    """Raw PageSpeed Insights API call — this is the function we cache."""
    params = {
        "url": url,
        "key": api_key,
        "strategy": strategy,
        "category": "performance",
    }
    response = requests.get(PAGESPEED_API_URL, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def _parse_pagespeed(data: dict) -> dict:
    """Extract Core Web Vitals and performance score from API response."""
    result = {
        "score": None,
        "lcp": None,
        "lcp_rating": None,
        "fid": None,
        "fid_rating": None,
        "cls": None,
        "cls_rating": None,
        "inp": None,
        "inp_rating": None,
        "fcp": None,
        "fcp_rating": None,
        "ttfb": None,
        "ttfb_rating": None,
    }

    # Lighthouse performance score (0-100)
    try:
        categories = data.get("lighthouseResult", {}).get("categories", {})
        perf = categories.get("performance", {})
        score_raw = perf.get("score")
        if score_raw is not None:
            result["score"] = int(score_raw * 100)
    except Exception:
        pass

    # Field data from Chrome UX Report (real user metrics)
    metrics = data.get("loadingExperience", {}).get("metrics", {})

    metric_map = {
        "LARGEST_CONTENTFUL_PAINT_MS": ("lcp", "lcp_rating", 0.001),  # ms -> seconds
        "FIRST_INPUT_DELAY_MS": ("fid", "fid_rating", 1.0),  # keep as ms
        "CUMULATIVE_LAYOUT_SHIFT_SCORE": ("cls", "cls_rating", 0.01),  # centis -> score
        "INTERACTION_TO_NEXT_PAINT": ("inp", "inp_rating", 1.0),  # ms
        "FIRST_CONTENTFUL_PAINT_MS": ("fcp", "fcp_rating", 0.001),  # ms -> seconds
        "EXPERIMENTAL_TIME_TO_FIRST_BYTE": ("ttfb", "ttfb_rating", 0.001),  # ms -> seconds
    }

    for api_key_name, (field, rating_field, multiplier) in metric_map.items():
        metric = metrics.get(api_key_name, {})
        if metric:
            percentile = metric.get("percentile")
            category = metric.get("category")
            if percentile is not None:
                result[field] = round(percentile * multiplier, 3)
            if category:
                result[rating_field] = category  # "FAST", "AVERAGE", "SLOW"

    # Fallback: if no field data, try lab data from Lighthouse
    if result["lcp"] is None:
        audits = data.get("lighthouseResult", {}).get("audits", {})
        lab_map = {
            "largest-contentful-paint": ("lcp", 0.001),
            "first-contentful-paint": ("fcp", 0.001),
            "cumulative-layout-shift": ("cls", 1.0),
            "total-blocking-time": ("fid", 1.0),  # TBT as proxy for FID
        }
        for audit_key, (field, multiplier) in lab_map.items():
            audit = audits.get(audit_key, {})
            val = audit.get("numericValue")
            if val is not None and result[field] is None:
                result[field] = round(val * multiplier, 3)

    return result


def get_pagespeed(url: str, strategy: str = "mobile") -> dict | None:
    """Get PageSpeed Insights data for a URL.

    Args:
        url: The page URL to analyse.
        strategy: "mobile" or "desktop".

    Returns:
        Dict with score, lcp, fid, cls, inp, fcp, ttfb and their ratings.
        None if API key is not configured.
    """
    if not settings.pagespeed_api_key:
        logger.warning("PAGESPEED_API_KEY not set — skipping page speed analysis")
        return None

    data = cached_call(_fetch_pagespeed, url, settings.pagespeed_api_key, strategy)
    return _parse_pagespeed(data)


def format_pagespeed_issues(ps: dict | None) -> list[str]:
    """Convert page speed data into human-readable issue strings."""
    if ps is None:
        return ["Page speed data unavailable (PAGESPEED_API_KEY not configured)"]

    issues: list[str] = []

    if ps.get("score") is not None and ps["score"] < 50:
        issues.append(f"Performance score is {ps['score']}/100 — needs significant improvement")
    elif ps.get("score") is not None and ps["score"] < 90:
        issues.append(f"Performance score is {ps['score']}/100 — room for improvement")

    thresholds = {
        "lcp": ("Largest Contentful Paint", 2.5, "s"),
        "fcp": ("First Contentful Paint", 1.8, "s"),
        "cls": ("Cumulative Layout Shift", 0.1, ""),
        "inp": ("Interaction to Next Paint", 200, "ms"),
        "ttfb": ("Time to First Byte", 0.8, "s"),
    }

    for key, (label, threshold, unit) in thresholds.items():
        val = ps.get(key)
        rating = ps.get(f"{key}_rating")
        if val is not None and (rating in ("AVERAGE", "SLOW") or val > threshold):
            issues.append(f"{label} is {val}{unit} (threshold: {threshold}{unit}) — rated {rating or 'NEEDS WORK'}")

    return issues
