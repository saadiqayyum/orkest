# SEO Agent

An AI-powered SEO audit agent built with **LangGraph**. It accepts a target domain and a list of keywords, then runs a 7-node pipeline that collects rankings, scrapes top competitors, audits the target page, computes competitor-grounded insights, and finds content/SERP gaps.

The agent exposes itself via the **LangGraph Agent Server** so external apps (e.g. the dashboard in `../dg-seo-agent-app`) can run audits over HTTP.

---

## Pipeline

```
check_rankings          SerpAPI rankings + SERP features + PAA
     ↓
analyse_competitors     Scrape top N competitor pages (H1/H2/H3, meta, word count, schema, links)
     ↓
audit_onpage            On-page SEO checks; persists scraped target-page data
     ↓
pagespeed               Google PageSpeed Insights → Core Web Vitals
     ↓
find_content_gaps       Missing H2/H3 topics (attributed to competitors) + PAA + SERP-feature gaps
     ↓
competitor_insights     Data-backed deltas vs competitors (word count, schema, headings, links, meta, title)
     ↓
audit_internal_links    Internal-link score + issues
```

The LLM synthesiser (`report/synthesiser.py`) is **not wired into the graph** in V1 — the structured `results` array carries everything the web dashboard needs. See [V1 / V2](#v1--v2) below.

---

## Prerequisites

- Python **3.11+** (3.11 is what `langgraph.json` pins)
- A SerpAPI key, a Google PageSpeed Insights key, and (optional) a Gemini key if you re-enable the synthesiser.

Copy `.env.example` to `.env` and fill in:

```env
SERPAPI_KEY=
PAGESPEED_API_KEY=
GOOGLE_API_KEY=        # Gemini, only needed for the synthesiser
TARGET_DOMAIN=         # Optional default
MAX_COMPETITORS=5
CACHE_TTL_HOURS=24
```

---

## Running as a LangGraph server (recommended)

This is how the `dg-seo-agent-app` dashboard talks to the agent.

```bash
pip install -U "langgraph-cli[inmem]" -e .
langgraph dev --allow-blocking
```

- Binds to `http://127.0.0.1:2024`
- `--allow-blocking` is required because `audit_onpage` runs `asyncio.run()` inside a sync node
- Assistant ID: `seo_agent` (declared in `langgraph.json`)

The Agent Server exposes `/assistants`, `/threads`, and `/runs` endpoints. From the web app, use `@langchain/langgraph-sdk`:

```ts
import { Client } from "@langchain/langgraph-sdk";

const client = new Client({ apiUrl: "http://127.0.0.1:2024" });
const thread = await client.threads.create();
const state = await client.runs.wait(thread.thread_id, "seo_agent", {
  input: {
    target_domain: "https://example.com",
    keywords: ["seo audit tool"],
    results: [],
    final_report: null,
    errors: [],
  },
});
```

`langgraph.json` is the config the CLI reads:

```json
{
  "dependencies": ["."],
  "graphs": { "seo_agent": "./seo_agent/graph.py:graph" },
  "env": ".env",
  "python_version": "3.11"
}
```

---

## Running as a CLI (standalone)

For local testing without the server — also runs the LLM synthesiser and writes a markdown report.

```bash
pip install -r requirements.txt
python -m seo_agent.main \
  --domain https://yoursite.com \
  --keywords "seo audit tool" "keyword research" \
  --output ./output
```

Outputs `output/report_data.json` + `output/seo_report.md`.

---

## State schema (key fields)

```python
class KeywordData(TypedDict):
    keyword: str
    your_rank: int | None
    your_url: str | None
    your_page: int | None
    top_competitors: list[CompetitorRanking]
    serp_features: SERPFeatures
    on_page_issues: list[str]
    missing_topics: list[MissingTopic]          # richer: kind + source_competitors + frequency
    backlink_gap: list[str]
    internal_link_score: float
    internal_link_issues: list[str]
    page_speed: dict | None
    your_page_data: CompetitorData | None       # target page, same shape as competitors
    raw_competitor_data: list[CompetitorData]
    competitor_insights: list[CompetitorInsight]  # data-backed deltas with per-competitor evidence
```

`MissingTopic.kind` is `"heading" | "paa" | "serp_feature"` so the UI can group/label each gap with source-competitor attribution.

`CompetitorInsight` carries `category`, `severity`, `your_value`, `competitor_avg`, `recommendation`, and an `evidence[]` array of per-competitor measurements.

---

## Project structure

```
dg-seo-agent/
├── langgraph.json              # LangGraph Agent Server config
├── pyproject.toml              # makes `.` pip-installable
├── requirements.txt            # dev/CLI deps (mirrored in pyproject)
├── .env / .env.example
├── seo_agent/
│   ├── graph.py                # build_graph() + module-level compiled `graph`
│   ├── state.py                # TypedDicts (SEOAgentState, KeywordData, etc.)
│   ├── config.py
│   ├── main.py                 # CLI entry point
│   ├── cache/manager.py        # disk-backed caching for external API calls
│   ├── tools/
│   │   ├── serp.py
│   │   ├── competitor.py
│   │   ├── onpage.py
│   │   ├── pagespeed.py
│   │   ├── content_gap.py
│   │   ├── competitor_insights.py
│   │   ├── internal_links.py
│   │   ├── backlinks.py        # disabled in V1 (no Moz/Ahrefs key)
│   │   └── robots_check.py
│   └── report/
│       ├── synthesiser.py      # Gemini markdown synthesiser (disabled in V1)
│       └── formatter.py        # writes JSON + markdown for CLI mode
├── tests/                      # pytest suite (134 tests)
└── seo_agent_spec.md           # original design spec
```

---

## Testing

```bash
pytest tests/
```

Live tests that hit real APIs (SerpAPI, PageSpeed, Gemini) auto-skip when keys aren't set.

---

## V1 / V2

### What's in V1

**Agent side:**
- 7-node LangGraph pipeline — rankings → competitors → on-page → page speed → content gaps → competitor insights → internal links
- Competitor-grounded insights with per-competitor evidence (not generic LLM advice)
- Richer `missing_topics` objects with source-competitor attribution
- `langgraph dev` deployment contract with `langgraph.json`, `pyproject.toml`, module-level `graph` export
- Structured output only — the web dashboard consumes `ReportData` directly

**Deferred to V2 (files kept, not wired):**
- **LLM synthesiser** (`report/synthesiser.py`) — produces a Gemini-authored markdown report + cross-keyword summary. Currently runs only in CLI mode via `main.py`. To re-enable inside the graph, add `synthesise_report_node` as the final node in `graph.py`.
- **Backlink gap analysis** (`tools/backlinks.py`) — requires Moz/Ahrefs API keys. Commented out of the graph wiring.

### What's planned for V2

1. **Re-enable the synthesiser in-graph** so the Agent Server returns `final_report` alongside `results`.
2. **Backlink tool** once Moz or Ahrefs credentials are available.
3. **Historical comparison** — the dashboard will compare a user's audits over time to show improvement after recommendations are implemented. The agent doesn't need changes for this (timestamps are handled by the web app's Mongo layer), but we may add a node that diffs the current run against the previous audit passed in as state.
4. **Streaming per-node progress** — expose incremental state via `runs.stream` so the dashboard can show which node is currently executing.
