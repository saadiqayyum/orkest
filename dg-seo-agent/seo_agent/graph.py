"""LangGraph pipeline definition — wires all SEO agent nodes together."""

import logging

from langgraph.graph import StateGraph, END

from seo_agent.state import SEOAgentState
from seo_agent.tools.serp import check_rankings_node
from seo_agent.tools.competitor import analyse_competitors_node
from seo_agent.tools.onpage import audit_onpage_node
from seo_agent.tools.pagespeed import get_pagespeed, format_pagespeed_issues
# from seo_agent.tools.backlinks import find_backlink_gaps_node  # Skipped: no Moz key
from seo_agent.tools.content_gap import find_content_gaps_node
from seo_agent.tools.competitor_insights import competitor_insights_node
from seo_agent.tools.internal_links import audit_internal_links_node

logger = logging.getLogger(__name__)


def _pagespeed_node(state: dict) -> dict:
    """Wrapper node: fetch PageSpeed data and append issues to on_page_issues."""
    results = state.get("results", [])
    errors = state.get("errors", [])
    target_domain = state["target_domain"]

    for kw_data in results:
        keyword = kw_data["keyword"]
        url = kw_data.get("your_url")

        if not url:
            if not target_domain.startswith("http"):
                url = f"https://{target_domain}"
            else:
                url = target_domain.rstrip("/")

        try:
            ps_data = get_pagespeed(url)
            kw_data["page_speed"] = ps_data
            ps_issues = format_pagespeed_issues(ps_data)
            kw_data["on_page_issues"].extend(ps_issues)
            logger.info("PageSpeed for '%s': score=%s", keyword,
                        ps_data.get("score") if ps_data else "N/A")
        except Exception as e:
            error_msg = f"[pagespeed] keyword='{keyword}' error: {e}"
            logger.error(error_msg)
            errors.append(error_msg)

    return {**state, "results": results, "errors": errors}


def build_graph() -> StateGraph:
    """Build and compile the SEO agent LangGraph pipeline.

    Pipeline:
        check_rankings -> analyse_competitors -> audit_onpage -> pagespeed
        -> find_backlink_gaps -> find_content_gaps -> audit_internal_links -> END

    Returns:
        Compiled StateGraph ready for invocation.
    """
    graph = StateGraph(SEOAgentState)

    # Add nodes
    graph.add_node("check_rankings", check_rankings_node)
    graph.add_node("analyse_competitors", analyse_competitors_node)
    graph.add_node("audit_onpage", audit_onpage_node)
    graph.add_node("pagespeed", _pagespeed_node)
    # graph.add_node("find_backlink_gaps", find_backlink_gaps_node)  # Skipped: no Moz key
    graph.add_node("find_content_gaps", find_content_gaps_node)
    graph.add_node("competitor_insights", competitor_insights_node)
    graph.add_node("audit_internal_links", audit_internal_links_node)

    # Wire edges (sequential pipeline)
    graph.set_entry_point("check_rankings")
    graph.add_edge("check_rankings", "analyse_competitors")
    graph.add_edge("analyse_competitors", "audit_onpage")
    graph.add_edge("audit_onpage", "pagespeed")
    # graph.add_edge("pagespeed", "find_backlink_gaps")  # Skipped: no Moz key
    # graph.add_edge("find_backlink_gaps", "find_content_gaps")
    graph.add_edge("pagespeed", "find_content_gaps")
    graph.add_edge("find_content_gaps", "competitor_insights")
    graph.add_edge("competitor_insights", "audit_internal_links")
    graph.add_edge("audit_internal_links", END)

    return graph.compile()


# Module-level compiled instance — referenced by langgraph.json as
# `./seo_agent/graph.py:graph` so `langgraph dev` can serve it.
graph = build_graph()
