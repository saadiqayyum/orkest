"""Report formatter — writes final outputs as Markdown and JSON files."""

import json
import logging
import os
from datetime import datetime, timezone

from seo_agent.state import SEOAgentState

logger = logging.getLogger(__name__)


def write_markdown(report: str | None, output_dir: str) -> str:
    """Write the final report as a Markdown file.

    Args:
        report: The full markdown report string.
        output_dir: Directory to write the file to.

    Returns:
        Path to the written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "seo_report.md")

    if not report:
        report = "# SEO Report\n\n*No report was generated.*"

    with open(path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info("Markdown report written to %s", path)
    return path


def _serialize_keyword_data(results: list[dict]) -> list[dict]:
    """Prepare keyword data for JSON serialization.

    Preserves every field the web-app needs, including raw_competitor_data,
    your_page_data, and the richer competitor_insights / missing_topics objects.
    """
    serializable = []
    for kw in results:
        entry = {
            "keyword": kw.get("keyword"),
            "your_rank": kw.get("your_rank"),
            "your_url": kw.get("your_url"),
            "your_page": kw.get("your_page"),
            "top_competitors": kw.get("top_competitors", []),
            "serp_features": kw.get("serp_features", {}),
            "on_page_issues": kw.get("on_page_issues", []),
            "missing_topics": kw.get("missing_topics", []),
            "backlink_gap": kw.get("backlink_gap", []),
            "internal_link_score": kw.get("internal_link_score", 0.0),
            "internal_link_issues": kw.get("internal_link_issues", []),
            "page_speed": kw.get("page_speed"),
            "your_page_data": kw.get("your_page_data"),
            "raw_competitor_data": kw.get("raw_competitor_data", []),
            "competitor_insights": kw.get("competitor_insights", []),
        }
        serializable.append(entry)
    return serializable


def write_json(results: list[dict], target_domain: str, output_dir: str) -> str:
    """Write structured keyword data as a JSON file.

    Args:
        results: List of KeywordData dicts.
        target_domain: The target domain.
        output_dir: Directory to write the file to.

    Returns:
        Path to the written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "report_data.json")

    data = {
        "domain": target_domain,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "keywords": _serialize_keyword_data(results),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("JSON data written to %s", path)
    return path
