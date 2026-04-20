"""Entry point for the SEO Agent — accepts keywords + domain, runs the pipeline."""

import argparse
import logging
import sys

from seo_agent.graph import build_graph
from seo_agent.report.synthesiser import synthesise_report_node
from seo_agent.report.formatter import write_markdown, write_json
from seo_agent.state import SEOAgentState


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the agent."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run(domain: str, keywords: list[str], output_dir: str = "./output") -> SEOAgentState:
    """Run the full SEO agent pipeline.

    Args:
        domain: Target domain (e.g. 'https://yoursite.com').
        keywords: List of keywords to analyse.
        output_dir: Directory for output files.

    Returns:
        Final pipeline state.
    """
    logger = logging.getLogger(__name__)

    # Normalize domain
    if not domain.startswith("http"):
        domain = f"https://{domain}"

    # Build and run the data collection pipeline
    logger.info("Starting SEO audit for %s with %d keywords", domain, len(keywords))
    logger.info("Keywords: %s", ", ".join(keywords))

    graph = build_graph()

    initial_state: SEOAgentState = {
        "target_domain": domain,
        "keywords": keywords,
        "results": [],
        "final_report": None,
        "errors": [],
    }

    # Run data collection pipeline (all nodes except report synthesis)
    state = graph.invoke(initial_state)
    logger.info("Data collection complete. %d keywords processed, %d errors.",
                len(state["results"]), len(state["errors"]))

    # Run report synthesis (separate so it's clear in logs)
    logger.info("Generating report with Gemini...")
    state = synthesise_report_node(state)

    # Write outputs
    md_path = write_markdown(state["final_report"], output_dir)
    json_path = write_json(state["results"], domain, output_dir)

    logger.info("Report saved to %s", md_path)
    logger.info("Data saved to %s", json_path)

    if state["errors"]:
        logger.warning("Completed with %d errors:", len(state["errors"]))
        for err in state["errors"]:
            logger.warning("  - %s", err)
    else:
        logger.info("Completed successfully with no errors.")

    return state


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SEO Agent — AI-powered SEO audit tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m seo_agent.main --domain https://yoursite.com --keywords "seo tools" "best seo software"
  python -m seo_agent.main --domain yoursite.com --keywords "keyword research" --output ./reports
        """,
    )
    parser.add_argument(
        "--domain", required=True,
        help="Target domain (e.g. https://yoursite.com)",
    )
    parser.add_argument(
        "--keywords", required=True, nargs="+",
        help="Keywords to analyse (space-separated, use quotes for multi-word)",
    )
    parser.add_argument(
        "--output", default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose/debug logging",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    try:
        state = run(args.domain, args.keywords, args.output)
        print(f"\nReport saved to {args.output}/seo_report.md")
        print(f"Data saved to {args.output}/report_data.json")
        if state["errors"]:
            print(f"\nCompleted with {len(state['errors'])} errors (see report for details)")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
