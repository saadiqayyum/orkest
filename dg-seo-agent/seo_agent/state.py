"""Shared state schema for the SEO Agent pipeline."""

from typing import TypedDict


class SERPFeatures(TypedDict):
    """SERP features detected for a keyword."""

    has_featured_snippet: bool
    has_knowledge_panel: bool
    has_video_carousel: bool
    has_image_pack: bool
    has_local_pack: bool
    people_also_ask: list[str]


class CompetitorData(TypedDict):
    """Scraped data for a single competitor page."""

    url: str
    title: str
    meta_description: str
    h1: list[str]
    h2: list[str]
    h3: list[str]
    word_count: int
    has_schema: bool
    internal_links: int
    external_links: int


class CompetitorRanking(TypedDict):
    """Ranking info for a competitor in SERP results."""

    rank: int
    url: str
    title: str
    domain: str


class MissingTopic(TypedDict):
    """A topic competitors cover that the target page does not.

    Richer than a plain string: carries attribution to specific competitors
    so the agent can justify every recommendation with real competitor data.
    """

    topic: str                       # normalized heading
    example_heading: str             # verbatim heading as it appears on a competitor
    level: str                       # "H2" or "H3"
    frequency: int                   # number of competitors covering the topic
    source_competitors: list[str]    # URLs of competitors that cover it
    kind: str                        # "heading" | "paa" | "serp_feature"


class InsightEvidence(TypedDict, total=False):
    """A single piece of per-competitor evidence backing an insight."""

    competitor_url: str
    competitor_domain: str
    value: str                       # stringified metric (e.g. "3200 words", "yes")


class CompetitorInsight(TypedDict):
    """A data-backed finding derived from comparing target to competitors.

    Every insight is grounded in concrete per-competitor measurements so
    downstream recommendations can cite real data instead of generic advice.
    """

    category: str                    # "word_count" | "schema" | "headings" | "meta" | "internal_links" | "external_links" | "title"
    observation: str                 # human-readable finding
    severity: str                    # "high" | "medium" | "low"
    recommendation: str              # concrete, competitor-grounded action
    your_value: str                  # stringified target metric (or "N/A")
    competitor_avg: str              # stringified competitor average
    evidence: list[InsightEvidence]  # per-competitor breakdown


class KeywordData(TypedDict):
    """All collected data for a single keyword."""

    keyword: str
    your_url: str | None
    your_rank: int | None  # None = not in top 100
    your_page: int | None
    top_competitors: list[CompetitorRanking]
    serp_features: SERPFeatures
    on_page_issues: list[str]
    missing_topics: list[MissingTopic]  # competitor-attributed topic gaps
    backlink_gap: list[str]             # Domains linking to competitors but not you
    internal_link_score: float          # 0.0 – 1.0
    internal_link_issues: list[str]
    page_speed: dict | None             # Core Web Vitals from PageSpeed API
    raw_competitor_data: list[CompetitorData]
    your_page_data: CompetitorData | None  # target page, same shape as competitors
    competitor_insights: list[CompetitorInsight]


class SEOAgentState(TypedDict):
    """Top-level state passed through the LangGraph pipeline."""

    target_domain: str
    keywords: list[str]
    results: list[KeywordData]
    final_report: str | None
    errors: list[str]
