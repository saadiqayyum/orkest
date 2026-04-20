"""Content/topic gap analyser — finds topics competitors cover that you don't.

Unlike a plain string list, every gap carries attribution to the specific
competitors that cover it, plus a verbatim example heading. That way the
downstream report can say "Ahrefs and Moz both have a 'Search Intent' H2 —
add one" instead of generic LLM advice.
"""

import logging
from collections import defaultdict
from urllib.parse import urlparse

from thefuzz import fuzz

from seo_agent.state import CompetitorData, MissingTopic, SERPFeatures

logger = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 70


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def find_missing_topics(
    your_headings: list[str],
    competitors: list[CompetitorData],
    similarity_threshold: int = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[MissingTopic]:
    """Find heading topics covered by 2+ competitors but not your page.

    Returns MissingTopic objects with per-competitor attribution so every
    recommendation can cite concrete evidence.
    """
    your_topics = [h.lower().strip() for h in your_headings if h.strip()]

    # topic_index[normalized] = {"level": "H2"|"H3", "sources": {url: verbatim_heading}}
    topic_index: dict[str, dict] = defaultdict(
        lambda: {"level": "", "sources": {}}
    )

    for comp in competitors:
        url = comp.get("url", "")
        if not url:
            continue
        # H2 headings carry more weight than H3
        for heading in comp.get("h2", []):
            normalized = heading.lower().strip()
            if not normalized:
                continue
            topic_index[normalized]["level"] = "H2"
            topic_index[normalized]["sources"].setdefault(url, heading)
        for heading in comp.get("h3", []):
            normalized = heading.lower().strip()
            if not normalized or normalized in topic_index and topic_index[normalized]["level"] == "H2":
                continue
            topic_index[normalized]["level"] = "H3"
            topic_index[normalized]["sources"].setdefault(url, heading)

    # Keep only topics in 2+ competitors that you don't cover
    missing: list[MissingTopic] = []
    for topic, data in sorted(
        topic_index.items(),
        key=lambda kv: len(kv[1]["sources"]),
        reverse=True,
    ):
        sources = data["sources"]
        if len(sources) < 2:
            continue

        matched = any(
            fuzz.token_sort_ratio(topic, yh) >= similarity_threshold
            for yh in your_topics
        )
        if matched:
            continue

        first_url = next(iter(sources))
        missing.append({
            "topic": topic,
            "example_heading": sources[first_url],
            "level": data["level"] or "H2",
            "frequency": len(sources),
            "source_competitors": list(sources.keys()),
            "kind": "heading",
        })

        if len(missing) >= 20:
            break

    return missing


def find_paa_gaps(
    your_headings: list[str],
    your_text: str,
    paa_questions: list[str],
    similarity_threshold: int = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[MissingTopic]:
    """Find PAA questions the target page doesn't address.

    PAA gaps have no source competitors (the signal is SERP-level, not
    competitor-page-level), so source_competitors is empty.
    """
    your_lower = [h.lower().strip() for h in your_headings]
    text_lower = your_text.lower()

    unanswered: list[MissingTopic] = []
    for question in paa_questions:
        q_lower = question.lower().strip()

        if any(fuzz.token_sort_ratio(q_lower, h) >= similarity_threshold for h in your_lower):
            continue

        stop_words = {
            "what", "how", "why", "when", "where", "which", "who",
            "is", "are", "do", "does", "can", "the", "a", "an",
            "in", "on", "of", "to", "for", "and", "or", "it",
            "much", "many", "should", "would", "could", "best",
            "most", "some", "any", "all", "very", "really",
        }
        key_terms = [
            w.strip("?.,!;:'\"()[]")
            for w in q_lower.split()
            if w.strip("?.,!;:'\"()[]") not in stop_words
            and len(w.strip("?.,!;:'\"()[]")) > 2
        ]

        unanswered_q = False
        if key_terms:
            terms_found = sum(1 for t in key_terms if t in text_lower)
            coverage = terms_found / len(key_terms)
            if coverage < 0.6:
                unanswered_q = True
        else:
            unanswered_q = True

        if unanswered_q:
            unanswered.append({
                "topic": question,
                "example_heading": question,
                "level": "H2",
                "frequency": 0,
                "source_competitors": [],
                "kind": "paa",
            })

    return unanswered


def find_serp_feature_opportunities(
    serp_features: SERPFeatures,
    your_data: CompetitorData | None,
    competitor_data: list[CompetitorData],
) -> list[MissingTopic]:
    """Identify SERP feature opportunities, attributed to competitor evidence."""
    opportunities: list[MissingTopic] = []

    # Featured snippet — check if you have FAQ-like headings
    if serp_features.get("has_featured_snippet"):
        has_faq = False
        if your_data:
            headings = " ".join(your_data.get("h2", []) + your_data.get("h3", []))
            has_faq = any(w in headings.lower() for w in ["faq", "question", "?"])
        if not has_faq:
            comp_with_faq = [
                c for c in competitor_data
                if any("?" in h or "faq" in h.lower()
                       for h in c.get("h2", []) + c.get("h3", []))
            ]
            opportunities.append({
                "topic": "featured snippet opportunity — add FAQ section",
                "example_heading": "Add a FAQ / direct-answer section",
                "level": "H2",
                "frequency": len(comp_with_faq),
                "source_competitors": [c["url"] for c in comp_with_faq],
                "kind": "serp_feature",
            })

    # Video carousel
    if serp_features.get("has_video_carousel"):
        opportunities.append({
            "topic": "video carousel opportunity — create video content",
            "example_heading": "Video carousel visible in SERP",
            "level": "H2",
            "frequency": 0,
            "source_competitors": [],
            "kind": "serp_feature",
        })

    # Schema gap
    competitors_with_schema = [c for c in competitor_data if c.get("has_schema")]
    your_has_schema = your_data.get("has_schema", False) if your_data else False
    if len(competitors_with_schema) >= 2 and not your_has_schema:
        opportunities.append({
            "topic": "schema markup gap — competitors use structured data",
            "example_heading": f"{len(competitors_with_schema)} competitors use JSON-LD",
            "level": "H2",
            "frequency": len(competitors_with_schema),
            "source_competitors": [c["url"] for c in competitors_with_schema],
            "kind": "serp_feature",
        })

    # Image pack
    if serp_features.get("has_image_pack"):
        opportunities.append({
            "topic": "image pack opportunity — optimize images",
            "example_heading": "Image pack visible in SERP",
            "level": "H2",
            "frequency": 0,
            "source_competitors": [],
            "kind": "serp_feature",
        })

    return opportunities


def find_content_gaps_node(state: dict) -> dict:
    """LangGraph node: find content gaps for each keyword."""
    results = state.get("results", [])
    errors = state.get("errors", [])

    for kw_data in results:
        keyword = kw_data["keyword"]

        try:
            competitors = kw_data.get("raw_competitor_data", [])

            your_data = kw_data.get("your_page_data")
            your_headings: list[str] = []
            your_text = ""
            if your_data:
                your_headings = (
                    your_data.get("h1", [])
                    + your_data.get("h2", [])
                    + your_data.get("h3", [])
                )
                your_text = " ".join(your_headings)

            # Competitor-attributed heading gaps
            missing = find_missing_topics(your_headings, competitors)

            # PAA gaps
            paa_questions = kw_data.get("serp_features", {}).get("people_also_ask", [])
            if paa_questions:
                missing.extend(find_paa_gaps(your_headings, your_text, paa_questions))

            # SERP feature opportunities
            serp_features = kw_data.get("serp_features", {})
            missing.extend(
                find_serp_feature_opportunities(serp_features, your_data, competitors)
            )

            kw_data["missing_topics"] = missing

            logger.info(
                "Content gaps for '%s': %d headings, %d PAA, %d SERP-feature",
                keyword,
                sum(1 for m in missing if m["kind"] == "heading"),
                sum(1 for m in missing if m["kind"] == "paa"),
                sum(1 for m in missing if m["kind"] == "serp_feature"),
            )

        except Exception as e:
            error_msg = f"[find_content_gaps] keyword='{keyword}' error: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            kw_data.setdefault("missing_topics", [])

    return {**state, "results": results, "errors": errors}
