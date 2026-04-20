"""LLM report writer — grounds every recommendation in concrete competitor data.

The prompt is structured so the model must cite a specific competitor URL or
metric for each suggestion. Generic "write better content" advice is explicitly
discouraged; the structured insights/missing_topics blocks are the source of truth.
"""

import logging

from google import genai

from seo_agent.config import settings
from seo_agent.state import KeywordData, MissingTopic, CompetitorInsight

logger = logging.getLogger(__name__)

KEYWORD_REPORT_PROMPT = """You are a senior SEO strategist. Write a report section for ONE keyword.

Every recommendation you make MUST cite a specific competitor URL or metric
from the data below. Do not write generic SEO advice. If the data does not
support a claim, do not make the claim.

Domain: {target_domain}
Keyword: "{keyword}"

## Data Snapshot

Current ranking: {your_rank} (URL: {your_url})
Page in SERP: {your_page}

Your page (scraped):
{your_page_data_formatted}

SERP Features:
{serp_features_formatted}

People Also Ask questions:
{paa_questions}

Top competitors (from SERP):
{competitors_formatted}

## Competitor-Grounded Insights (each has per-competitor evidence)

{competitor_insights_formatted}

## Missing Topics (with source-competitor attribution)

{missing_topics_formatted}

## On-Page Issues Detected

{on_page_issues}

## Page Speed (Core Web Vitals)

{page_speed_formatted}

## Internal Linking

Score: {internal_link_score}/1.0
Issues:
{internal_link_issues}

## Backlink Gaps

{backlink_gaps}

## Required Report Sections

Write the following sections using the data above. **Every bullet must cite
a competitor URL, domain, or metric.** Do not invent data.

### Ranking Summary
How far from page 1, whether rank justifies effort.

### SERP Landscape
Which SERP features exist and which ones we can realistically target
(cite competitor structure from the insights).

### Why Competitors Outrank Us
Specific deltas from the Competitor-Grounded Insights block. For each point,
cite the competitor domain and metric (e.g. "ahrefs.com has 3200 words vs our 1200").

### Content Gaps to Close
Walk through the Missing Topics block. For each missing heading, name the
competitors that cover it and propose the exact H2/H3 to add. Do the same for
any PAA questions flagged as unanswered.

### On-Page Fixes (prioritised)
Order by severity. Each fix cites a measurement from the data.

### Internal Linking Fixes
Cite the internal-link score and specific linking issues.

### Action Plan
- **Quick wins (< 1 day):** e.g. title rewrite, meta description, schema snippet
- **Medium-term (1-4 weeks):** content expansion, new sections from missing topics
- **Long-term:** authority building (only if backlink data is present)

Write in clean markdown. Be specific. Be direct. Cite.
"""

SUMMARY_PROMPT = """You are a senior SEO strategist. Below are individual keyword reports for the domain {target_domain}.

Write a **Cross-Keyword Summary** that:
1. Identifies the top 10 highest-impact actions across ALL keywords
2. Groups common issues (e.g. if multiple keywords have the same on-page problem)
3. Prioritises actions by estimated impact (high/medium/low)
4. Provides a final recommended execution order

Every item must cite a concrete data point (keyword name, metric, or competitor).
Be concise and actionable. Use markdown formatting.

## Individual Keyword Reports

{keyword_reports}
"""


def _get_client() -> genai.Client:
    """Create a Gemini client."""
    return genai.Client(api_key=settings.google_api_key)


def _format_serp_features(kw: KeywordData) -> str:
    sf = kw.get("serp_features", {})
    lines = []
    features = {
        "has_featured_snippet": "Featured Snippet",
        "has_knowledge_panel": "Knowledge Panel",
        "has_video_carousel": "Video Carousel",
        "has_image_pack": "Image Pack",
        "has_local_pack": "Local Pack",
    }
    for key, label in features.items():
        if sf.get(key):
            lines.append(f"  - {label}: YES")
    if not lines:
        lines.append("  - No special SERP features detected")
    return "\n".join(lines)


def _format_paa(kw: KeywordData) -> str:
    paa = kw.get("serp_features", {}).get("people_also_ask", [])
    if not paa:
        return "  - None found"
    return "\n".join(f"  - {q}" for q in paa)


def _format_competitors(kw: KeywordData) -> str:
    comps = kw.get("top_competitors", [])
    if not comps:
        return "  - No competitors found"
    lines = []
    for c in comps[:10]:
        lines.append(f"  - #{c['rank']} {c['domain']} — \"{c['title']}\" ({c['url']})")
    return "\n".join(lines)


def _format_issues(issues: list[str]) -> str:
    if not issues:
        return "  - None found"
    return "\n".join(f"  - {issue}" for issue in issues)


def _format_page_speed(kw: KeywordData) -> str:
    ps = kw.get("page_speed")
    if not ps:
        return "  - No page speed data available"
    lines = []
    if ps.get("score") is not None:
        lines.append(f"  - Performance Score: {ps['score']}/100")
    for key, label in [("lcp", "LCP"), ("fcp", "FCP"), ("cls", "CLS"), ("inp", "INP"), ("ttfb", "TTFB")]:
        val = ps.get(key)
        rating = ps.get(f"{key}_rating", "")
        if val is not None:
            unit = "ms" if key == "inp" else ("" if key == "cls" else "s")
            lines.append(f"  - {label}: {val}{unit} ({rating})")
    return "\n".join(lines) if lines else "  - No page speed data available"


def _format_your_page_data(kw: KeywordData) -> str:
    p = kw.get("your_page_data")
    if not p:
        return "  - Target page data not available"
    return (
        f"  - Title: \"{p.get('title', '')}\" ({len(p.get('title', ''))} chars)\n"
        f"  - Meta: \"{p.get('meta_description', '')[:120]}\" ({len(p.get('meta_description', ''))} chars)\n"
        f"  - H1: {p.get('h1', [])}\n"
        f"  - H2 ({len(p.get('h2', []))}): {p.get('h2', [])[:8]}\n"
        f"  - H3 ({len(p.get('h3', []))}): {p.get('h3', [])[:8]}\n"
        f"  - Word count: {p.get('word_count', 0)}\n"
        f"  - Schema markup: {'yes' if p.get('has_schema') else 'no'}\n"
        f"  - Internal links: {p.get('internal_links', 0)} | External: {p.get('external_links', 0)}"
    )


def _format_competitor_insights(insights: list[CompetitorInsight]) -> str:
    if not insights:
        return "  - No competitor-grounded insights generated"
    lines = []
    for i in insights:
        lines.append(
            f"- **[{i['severity'].upper()}] {i['category']}**: {i['observation']}\n"
            f"  - You: {i['your_value']} | Competitors: {i['competitor_avg']}\n"
            f"  - Recommendation: {i['recommendation']}\n"
            f"  - Evidence: "
            + "; ".join(
                f"{e.get('competitor_domain', '')} = {e.get('value', '')}"
                for e in i.get("evidence", [])[:5]
            )
        )
    return "\n".join(lines)


def _format_missing_topics(topics: list[MissingTopic]) -> str:
    if not topics:
        return "  - No content gaps detected"

    by_kind: dict[str, list[MissingTopic]] = {"heading": [], "paa": [], "serp_feature": []}
    for t in topics:
        by_kind.setdefault(t.get("kind", "heading"), []).append(t)

    lines = []
    if by_kind["heading"]:
        lines.append("**Heading gaps (competitors cover, you don't):**")
        for t in by_kind["heading"][:15]:
            sources = ", ".join(t.get("source_competitors", [])[:3])
            lines.append(
                f"  - [{t['level']}] \"{t['example_heading']}\" "
                f"({t['frequency']} competitors) — sources: {sources}"
            )
    if by_kind["paa"]:
        lines.append("\n**Unanswered PAA questions:**")
        for t in by_kind["paa"]:
            lines.append(f"  - {t['topic']}")
    if by_kind["serp_feature"]:
        lines.append("\n**SERP-feature opportunities:**")
        for t in by_kind["serp_feature"]:
            ref = ""
            if t.get("source_competitors"):
                ref = f" (seen on {len(t['source_competitors'])} competitors)"
            lines.append(f"  - {t['topic']}{ref}")

    return "\n".join(lines)


def generate_keyword_report(kw: KeywordData, target_domain: str) -> str:
    """Generate an LLM report for a single keyword using Gemini."""
    prompt = KEYWORD_REPORT_PROMPT.format(
        target_domain=target_domain,
        keyword=kw["keyword"],
        your_rank=kw.get("your_rank") or "Not in top 100",
        your_url=kw.get("your_url") or "N/A (not ranking)",
        your_page=kw.get("your_page") or "N/A",
        your_page_data_formatted=_format_your_page_data(kw),
        serp_features_formatted=_format_serp_features(kw),
        paa_questions=_format_paa(kw),
        competitors_formatted=_format_competitors(kw),
        competitor_insights_formatted=_format_competitor_insights(kw.get("competitor_insights", [])),
        missing_topics_formatted=_format_missing_topics(kw.get("missing_topics", [])),
        on_page_issues=_format_issues(kw.get("on_page_issues", [])),
        page_speed_formatted=_format_page_speed(kw),
        internal_link_score=kw.get("internal_link_score", 0.0),
        internal_link_issues=_format_issues(kw.get("internal_link_issues", [])),
        backlink_gaps=_format_issues(kw.get("backlink_gap", [])),
    )

    client = _get_client()
    response = client.models.generate_content(
        model=settings.gemini_model_id,
        contents=prompt,
    )

    return response.text


def generate_summary_report(keyword_reports: list[str], target_domain: str) -> str:
    """Generate a cross-keyword summary report using Gemini."""
    combined = "\n\n---\n\n".join(keyword_reports)

    prompt = SUMMARY_PROMPT.format(
        target_domain=target_domain,
        keyword_reports=combined,
    )

    client = _get_client()
    response = client.models.generate_content(
        model=settings.gemini_model_id,
        contents=prompt,
    )

    return response.text


def synthesise_report_node(state: dict) -> dict:
    """LangGraph node: generate the final SEO report using Gemini.

    Generates one report section per keyword, then a cross-keyword summary.
    """
    results = state.get("results", [])
    errors = state.get("errors", [])
    target_domain = state["target_domain"]

    if not settings.google_api_key:
        logger.error("GOOGLE_API_KEY not set — cannot generate report")
        errors.append("[synthesise_report] GOOGLE_API_KEY not configured")
        return {**state, "errors": errors}

    keyword_reports: list[str] = []

    for kw_data in results:
        keyword = kw_data["keyword"]
        try:
            logger.info("Generating report for keyword '%s'...", keyword)
            report = generate_keyword_report(kw_data, target_domain)
            keyword_reports.append(f"# Keyword: \"{keyword}\"\n\n{report}")
            logger.info("Report generated for '%s' (%d chars)", keyword, len(report))
        except Exception as e:
            error_msg = f"[synthesise_report] keyword='{keyword}' error: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            keyword_reports.append(
                f"# Keyword: \"{keyword}\"\n\n*Report generation failed: {e}*"
            )

    summary = ""
    if len(keyword_reports) > 1:
        try:
            logger.info("Generating cross-keyword summary...")
            summary = generate_summary_report(keyword_reports, target_domain)
            summary = f"# Cross-Keyword Summary\n\n{summary}"
        except Exception as e:
            error_msg = f"[synthesise_report] summary error: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            summary = "# Cross-Keyword Summary\n\n*Summary generation failed.*"

    header = f"# SEO Audit Report — {target_domain}\n\n"
    final_report = header + "\n\n---\n\n".join(keyword_reports)
    if summary:
        final_report += f"\n\n---\n\n{summary}"

    if errors:
        final_report += "\n\n---\n\n# Errors\n\n"
        final_report += "\n".join(f"- {e}" for e in errors)

    return {**state, "final_report": final_report, "errors": errors}
