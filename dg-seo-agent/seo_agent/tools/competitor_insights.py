"""Competitor-grounded insights — data-backed deltas vs the top-ranking pages.

Every insight here is derived from actual scraped competitor data and carries
per-competitor evidence so downstream recommendations can cite concrete facts
rather than generic SEO advice.
"""

import logging
from urllib.parse import urlparse

from seo_agent.state import CompetitorData, CompetitorInsight, InsightEvidence

logger = logging.getLogger(__name__)


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _avg(values: list[float]) -> float:
    clean = [v for v in values if v is not None and v > 0]
    return sum(clean) / len(clean) if clean else 0.0


def _evidence(competitors: list[CompetitorData], value_fn) -> list[InsightEvidence]:
    """Build per-competitor evidence rows using a value extractor."""
    rows: list[InsightEvidence] = []
    for c in competitors:
        if not c.get("url") or (c.get("word_count", 0) == 0 and not c.get("title")):
            continue  # skip blocked / failed scrapes
        rows.append({
            "competitor_url": c.get("url", ""),
            "competitor_domain": _domain(c.get("url", "")),
            "value": str(value_fn(c)),
        })
    return rows


def _word_count_insight(
    your_data: CompetitorData | None, competitors: list[CompetitorData]
) -> CompetitorInsight | None:
    comp_counts = [c.get("word_count", 0) for c in competitors]
    avg = _avg(comp_counts)
    if avg == 0:
        return None

    your_wc = your_data.get("word_count", 0) if your_data else 0
    gap = int(avg - your_wc)
    pct = (your_wc / avg * 100) if avg else 0

    if your_wc == 0 or your_wc < avg * 0.8:
        severity = "high" if pct < 50 else "medium"
        return {
            "category": "word_count",
            "observation": (
                f"Your page has {your_wc} words; top competitors average "
                f"{int(avg)} words ({pct:.0f}% of competitor depth)."
            ),
            "severity": severity,
            "recommendation": (
                f"Expand content by ~{gap} words. Match the depth of "
                f"competitors like {_domain(competitors[0]['url'])} "
                f"({competitors[0].get('word_count', 0)} words)."
                if competitors else f"Expand content by ~{gap} words."
            ),
            "your_value": f"{your_wc} words",
            "competitor_avg": f"{int(avg)} words",
            "evidence": _evidence(competitors, lambda c: f"{c.get('word_count', 0)} words"),
        }
    return None


def _schema_insight(
    your_data: CompetitorData | None, competitors: list[CompetitorData]
) -> CompetitorInsight | None:
    with_schema = [c for c in competitors if c.get("has_schema")]
    total = len([c for c in competitors if c.get("url")])
    if total == 0:
        return None

    your_has = your_data.get("has_schema", False) if your_data else False
    count = len(with_schema)

    if count >= 2 and not your_has:
        domains = ", ".join(sorted({_domain(c["url"]) for c in with_schema[:3]}))
        return {
            "category": "schema",
            "observation": (
                f"{count}/{total} competitors use JSON-LD schema markup; your page has none."
            ),
            "severity": "high" if count == total else "medium",
            "recommendation": (
                f"Add JSON-LD structured data (Article/FAQ/Product as appropriate). "
                f"Follow the pattern used by {domains}."
            ),
            "your_value": "no schema",
            "competitor_avg": f"{count}/{total} have schema",
            "evidence": _evidence(
                competitors,
                lambda c: "has schema" if c.get("has_schema") else "no schema",
            ),
        }
    return None


def _heading_depth_insight(
    your_data: CompetitorData | None, competitors: list[CompetitorData]
) -> CompetitorInsight | None:
    comp_h2 = [len(c.get("h2", [])) for c in competitors]
    comp_h3 = [len(c.get("h3", [])) for c in competitors]
    avg_h2 = _avg(comp_h2)
    avg_h3 = _avg(comp_h3)
    if avg_h2 == 0 and avg_h3 == 0:
        return None

    your_h2 = len(your_data.get("h2", [])) if your_data else 0
    your_h3 = len(your_data.get("h3", [])) if your_data else 0

    h2_gap = avg_h2 - your_h2
    h3_gap = avg_h3 - your_h3

    if h2_gap < 2 and h3_gap < 2:
        return None

    top = competitors[0] if competitors else None
    top_ref = ""
    if top:
        top_ref = (
            f" Top result ({_domain(top['url'])}) uses {len(top.get('h2', []))} H2s "
            f"and {len(top.get('h3', []))} H3s."
        )

    return {
        "category": "headings",
        "observation": (
            f"Your structure: {your_h2} H2 / {your_h3} H3. "
            f"Competitor average: {avg_h2:.1f} H2 / {avg_h3:.1f} H3."
        ),
        "severity": "medium" if (h2_gap >= 2 or h3_gap >= 3) else "low",
        "recommendation": (
            f"Restructure the page with more sub-sections.{top_ref}"
        ),
        "your_value": f"{your_h2} H2 / {your_h3} H3",
        "competitor_avg": f"{avg_h2:.1f} H2 / {avg_h3:.1f} H3",
        "evidence": _evidence(
            competitors,
            lambda c: f"{len(c.get('h2', []))} H2 / {len(c.get('h3', []))} H3",
        ),
    }


def _internal_links_insight(
    your_data: CompetitorData | None, competitors: list[CompetitorData]
) -> CompetitorInsight | None:
    avg = _avg([c.get("internal_links", 0) for c in competitors])
    if avg == 0:
        return None
    your_links = your_data.get("internal_links", 0) if your_data else 0

    if your_links >= avg * 0.7:
        return None

    gap = int(avg - your_links)
    return {
        "category": "internal_links",
        "observation": (
            f"Your page has {your_links} internal links; competitors average {int(avg)}."
        ),
        "severity": "medium",
        "recommendation": (
            f"Add ~{gap} contextual internal links to related pages on your domain. "
            f"Competitors use internal linking to spread PageRank across topical clusters."
        ),
        "your_value": f"{your_links} internal links",
        "competitor_avg": f"{int(avg)} internal links",
        "evidence": _evidence(
            competitors, lambda c: f"{c.get('internal_links', 0)} internal links"
        ),
    }


def _external_links_insight(
    your_data: CompetitorData | None, competitors: list[CompetitorData]
) -> CompetitorInsight | None:
    avg = _avg([c.get("external_links", 0) for c in competitors])
    if avg < 2:
        return None
    your_links = your_data.get("external_links", 0) if your_data else 0

    if your_links >= max(2, avg * 0.5):
        return None

    return {
        "category": "external_links",
        "observation": (
            f"Your page has {your_links} outbound citations; competitors average {int(avg)}."
        ),
        "severity": "low",
        "recommendation": (
            "Cite authoritative external sources to build topical credibility. "
            "Aim for 3-5 high-quality outbound links to primary sources."
        ),
        "your_value": f"{your_links} external links",
        "competitor_avg": f"{int(avg)} external links",
        "evidence": _evidence(
            competitors, lambda c: f"{c.get('external_links', 0)} external links"
        ),
    }


def _meta_description_insight(
    your_data: CompetitorData | None, competitors: list[CompetitorData]
) -> CompetitorInsight | None:
    comp_metas = [c.get("meta_description", "") for c in competitors if c.get("meta_description")]
    if len(comp_metas) < 2:
        return None

    your_meta = (your_data.get("meta_description", "") if your_data else "").strip()
    avg_len = _avg([len(m) for m in comp_metas])

    if your_meta and abs(len(your_meta) - avg_len) < 30:
        return None

    return {
        "category": "meta",
        "observation": (
            f"Your meta description: {len(your_meta)} chars. "
            f"Competitors average {int(avg_len)} chars across {len(comp_metas)} pages."
        ),
        "severity": "medium" if not your_meta else "low",
        "recommendation": (
            "Write a 150-160 char meta description that leads with the target keyword "
            "and includes a concrete value proposition. "
            f"Example competitor copy: \"{comp_metas[0][:120]}...\""
        ),
        "your_value": f"{len(your_meta)} chars" if your_meta else "missing",
        "competitor_avg": f"{int(avg_len)} chars",
        "evidence": _evidence(
            competitors, lambda c: f"{len(c.get('meta_description', ''))} chars"
        ),
    }


def _title_keyword_insight(
    keyword: str,
    your_data: CompetitorData | None,
    competitors: list[CompetitorData],
) -> CompetitorInsight | None:
    kw_lower = keyword.lower().strip()
    comp_with_kw = [c for c in competitors if kw_lower in (c.get("title") or "").lower()]
    total = len([c for c in competitors if c.get("title")])
    if total == 0:
        return None

    your_title = (your_data.get("title", "") if your_data else "").lower()
    your_has_kw = kw_lower in your_title

    if your_has_kw or len(comp_with_kw) < 2:
        return None

    examples = [f'"{c["title"]}"' for c in comp_with_kw[:2]]
    return {
        "category": "title",
        "observation": (
            f"{len(comp_with_kw)}/{total} competitors include \"{keyword}\" in their title; yours does not."
        ),
        "severity": "high",
        "recommendation": (
            f"Rewrite your title tag to include the phrase \"{keyword}\" near the start. "
            f"Competitor patterns: {', '.join(examples)}."
        ),
        "your_value": your_data.get("title", "") if your_data else "",
        "competitor_avg": f"{len(comp_with_kw)}/{total} contain the keyword",
        "evidence": _evidence(
            competitors,
            lambda c: "contains keyword" if kw_lower in (c.get("title") or "").lower() else "missing keyword",
        ),
    }


INSIGHT_BUILDERS = [
    _word_count_insight,
    _schema_insight,
    _heading_depth_insight,
    _internal_links_insight,
    _external_links_insight,
    _meta_description_insight,
]


def build_insights(
    keyword: str,
    your_data: CompetitorData | None,
    competitors: list[CompetitorData],
) -> list[CompetitorInsight]:
    """Run every insight builder and return non-None results."""
    # Filter out failed scrapes so averages aren't skewed
    valid_competitors = [
        c for c in competitors
        if c.get("url") and c.get("word_count", 0) > 0
    ]

    insights: list[CompetitorInsight] = []

    title_kw = _title_keyword_insight(keyword, your_data, valid_competitors)
    if title_kw:
        insights.append(title_kw)

    for builder in INSIGHT_BUILDERS:
        insight = builder(your_data, valid_competitors)
        if insight:
            insights.append(insight)

    severity_order = {"high": 0, "medium": 1, "low": 2}
    insights.sort(key=lambda i: severity_order.get(i["severity"], 3))
    return insights


def competitor_insights_node(state: dict) -> dict:
    """LangGraph node: compute competitor-grounded insights for each keyword."""
    results = state.get("results", [])
    errors = state.get("errors", [])

    for kw_data in results:
        keyword = kw_data["keyword"]
        try:
            insights = build_insights(
                keyword,
                kw_data.get("your_page_data"),
                kw_data.get("raw_competitor_data", []),
            )
            kw_data["competitor_insights"] = insights
            logger.info(
                "Competitor insights for '%s': %d findings (%s)",
                keyword,
                len(insights),
                ", ".join(i["category"] for i in insights),
            )
        except Exception as e:
            error_msg = f"[competitor_insights] keyword='{keyword}' error: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            kw_data.setdefault("competitor_insights", [])

    return {**state, "results": results, "errors": errors}
