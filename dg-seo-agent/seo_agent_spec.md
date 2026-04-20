# SEO Agent — Project Specification

## Overview

Build an AI-powered SEO agent using **LangGraph** (LangChain ecosystem) that accepts a list of keywords and a target domain, then produces a complete SEO audit report with competitor analysis, gap identification, and prioritised improvement suggestions.

The agent runs a structured pipeline: for each keyword it checks rankings, scrapes top competitors, audits the target page, identifies content and backlink gaps, then passes all structured data to an LLM which writes the final report.

---

## Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| Agent framework | LangGraph | Predictable pipeline, easy to debug and retry individual nodes |
| LLM | Claude claude-sonnet-4-6 via Anthropic SDK | Report writing and synthesis |
| SERP data | SerpAPI | Simple REST API, reliable ranking data + PAA + SERP features |
| Backlink data | Moz API (free tier) or Ahrefs API | Backlink gap analysis |
| Web scraping | `httpx` + `BeautifulSoup4` | Competitor page analysis |
| Page speed | Google PageSpeed Insights API | Core Web Vitals (free, 25k queries/day) |
| Topic matching | `thefuzz` | Lightweight fuzzy heading comparison — edge cases handled by LLM synthesis |
| Caching | Redis (or fallback: `diskcache`) | Avoid redundant API calls, respect rate limits |
| Output | Markdown report + JSON data file | Human-readable + machine-readable |
| Config | `.env` + `pydantic-settings` | Manage all API keys cleanly |

---

## Project Structure

```
seo-agent/
├── main.py                  # Entry point — accepts keywords + domain
├── graph.py                 # LangGraph pipeline definition
├── state.py                 # Shared state schema (TypedDict)
├── config.py                # Settings loaded from .env
├── tools/
│   ├── __init__.py
│   ├── serp.py              # SERP rank checker + PAA + SERP features
│   ├── competitor.py        # Competitor scraper + analyser
│   ├── onpage.py            # On-page SEO auditor
│   ├── backlinks.py         # Backlink gap tool
│   ├── content_gap.py       # Content/topic gap analyser (thefuzz)
│   ├── internal_links.py    # Internal link auditor
│   ├── pagespeed.py         # Google PageSpeed Insights integration
│   └── robots_check.py     # robots.txt compliance checker
├── report/
│   ├── __init__.py
│   ├── synthesiser.py       # LLM report writer
│   └── formatter.py         # Formats final Markdown output
├── cache/
│   └── manager.py           # Redis/diskcache abstraction
├── tests/
│   ├── test_tools.py
│   └── test_graph.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## Environment Variables

```env
# .env.example

ANTHROPIC_API_KEY=
SERPAPI_KEY=
MOZ_ACCESS_ID=
MOZ_SECRET_KEY=
AHREFS_API_KEY=           # Optional, preferred over Moz if available

TARGET_DOMAIN=            # e.g. https://yoursite.com
REDIS_URL=redis://localhost:6379  # Optional, falls back to diskcache

MAX_COMPETITORS=5         # How many top competitors to analyse per keyword
CACHE_TTL_HOURS=24        # How long to cache API responses
PAGESPEED_API_KEY=        # Google PageSpeed Insights API key (free)
```

---

## State Schema (`state.py`)

```python
from typing import TypedDict, Optional

class SERPFeatures(TypedDict):
    has_featured_snippet: bool
    has_knowledge_panel: bool
    has_video_carousel: bool
    has_image_pack: bool
    has_local_pack: bool
    people_also_ask: list[str]

class KeywordData(TypedDict):
    keyword: str
    your_url: str | None
    your_rank: int | None          # None = not in top 100
    your_page: int | None
    top_competitors: list[dict]    # [{url, rank, title, domain}]
    serp_features: SERPFeatures    # What SERP features exist for this keyword
    on_page_issues: list[str]
    missing_topics: list[str]
    backlink_gap: list[str]        # Domains linking to competitors but not you
    internal_link_score: float     # 0.0 – 1.0
    internal_link_issues: list[str]
    page_speed: dict | None        # Core Web Vitals from PageSpeed API
    raw_competitor_data: list[dict]

class SEOAgentState(TypedDict):
    target_domain: str
    keywords: list[str]
    results: list[KeywordData]
    final_report: str | None
    errors: list[str]
```

---

## Pipeline Nodes (`graph.py`)

The LangGraph pipeline has **7 sequential nodes**. Each node processes all keywords in the current batch.

```
check_rankings  (+ extract PAA, SERP features)
     ↓
analyse_competitors  (parallel per URL, with semaphore)
     ↓
audit_onpage  (+ page speed via PageSpeed API)
     ↓
find_backlink_gaps
     ↓
find_content_gaps  (thefuzz matching, + SERP feature gap analysis)
     ↓
audit_internal_links
     ↓
synthesise_report
```

### Node 1 — `check_rankings`

**File:** `tools/serp.py`

**What it does:**
- Calls SerpAPI for each keyword
- Finds target domain in the top 100 results
- Records: rank position, page number, result URL, result title
- Records top N competitors (configurable via `MAX_COMPETITORS`) — always collects top N regardless of whether target is ranking or not
- Extracts SERP features: featured snippets, knowledge panels, video carousels, image packs, local packs
- Extracts "People Also Ask" questions from SerpAPI response

**SerpAPI call:**
```python
import requests

def check_serp_ranking(keyword: str, target_domain: str, api_key: str) -> dict:
    params = {
        "q": keyword,
        "num": 100,
        "api_key": api_key,
        "engine": "google"
    }
    response = requests.get("https://serpapi.com/search", params=params)
    data = response.json()
    
    results = data.get("organic_results", [])
    
    your_position = None
    your_url = None
    competitors = []
    
    for i, result in enumerate(results):
        url = result.get("link", "")
        if target_domain in url:
            your_position = i + 1
            your_url = url
        elif len(competitors) < MAX_COMPETITORS:
            competitors.append({
                "rank": i + 1,
                "url": url,
                "title": result.get("title", ""),
                "domain": extract_domain(url)
            })
    
    # Extract SERP features
    serp_features = {
        "has_featured_snippet": "answer_box" in data or "featured_snippet" in data,
        "has_knowledge_panel": "knowledge_graph" in data,
        "has_video_carousel": "inline_videos" in data,
        "has_image_pack": "inline_images" in data,
        "has_local_pack": "local_results" in data,
        "people_also_ask": [
            q["question"] for q in data.get("related_questions", [])
        ]
    }
    
    return {
        "keyword": keyword,
        "your_rank": your_position,
        "your_url": your_url,
        "top_competitors": competitors,
        "serp_features": serp_features
    }
```

**Output:** Rank position, competitor URLs, SERP features, and PAA questions for each keyword.

---

### Node 2 — `analyse_competitors`

**File:** `tools/competitor.py`

**What it does:**
- Checks `robots.txt` before fetching (via `tools/robots_check.py`)
- Fetches each competitor URL with `httpx` (max 3 concurrent via `asyncio.Semaphore`)
- Parses with BeautifulSoup
- Extracts: title tag, meta description, H1/H2/H3 headings, word count, keyword density, schema markup types, link counts

```python
import httpx
import asyncio
from bs4 import BeautifulSoup

SCRAPE_SEMAPHORE = asyncio.Semaphore(3)

async def scrape_page(url: str) -> dict:
    async with SCRAPE_SEMAPHORE:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; SEOBot/1.0)"}
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=15, follow_redirects=True)
        soup = BeautifulSoup(response.text, "html.parser")
        
        return {
            "url": url,
            "title": soup.find("title").get_text() if soup.find("title") else "",
            "meta_description": get_meta_description(soup),
            "h1": [h.get_text().strip() for h in soup.find_all("h1")],
            "h2": [h.get_text().strip() for h in soup.find_all("h2")],
            "h3": [h.get_text().strip() for h in soup.find_all("h3")],
            "word_count": len(soup.get_text().split()),
            "has_schema": bool(soup.find("script", {"type": "application/ld+json"})),
            "internal_links": len([a for a in soup.find_all("a", href=True) 
                                   if extract_domain(a["href"]) == extract_domain(url)]),
            "external_links": len([a for a in soup.find_all("a", href=True) 
                                   if extract_domain(a["href"]) != extract_domain(url)])
        }
```

**Output:** Structured competitor profile per URL.

---

### Node 3 — `audit_onpage`

**File:** `tools/onpage.py`

**What it does:**
Audits the target domain's ranking page (or homepage if not ranking) for the keyword. Checks:

| Check | Pass Condition |
|---|---|
| Title tag exists | Yes |
| Title length | 50–60 characters |
| Keyword in title | Yes |
| Meta description exists | Yes |
| Meta description length | 150–160 characters |
| Single H1 | Exactly one H1 |
| Keyword in H1 | Yes |
| Word count vs competitors | Within 80% of top competitor average |
| Schema markup | At least one type present |
| Image alt tags | All images have alt text |
| Canonical tag | Present |

Also calls **Google PageSpeed Insights API** (`tools/pagespeed.py`) to get real Core Web Vitals:
- Largest Contentful Paint (LCP)
- First Input Delay (FID) / Interaction to Next Paint (INP)
- Cumulative Layout Shift (CLS)
- Overall performance score

**Output:** List of on-page issues + page speed data.

---

### Node 4 — `find_backlink_gaps`

**File:** `tools/backlinks.py`

**What it does:**
- Calls Moz or Ahrefs API to fetch the backlink profiles of top 3 competitors
- Fetches your own backlink profile
- Finds domains linking to 2+ competitors but NOT linking to you
- Sorts by domain authority (highest first)

**Output:** List of high-authority domains you should target for backlinks.

---

### Node 5 — `find_content_gaps`

**File:** `tools/content_gap.py`

**What it does:**
- Compares your page's H2/H3 headings against the union of all competitor H2/H3 headings
- Uses `thefuzz` for fuzzy heading matching (threshold: 70% similarity)
- Edge cases (semantically similar but differently worded headings) are handled by the LLM in the synthesis step
- Also checks for: FAQ sections, comparison tables
- Cross-references with "People Also Ask" data from Node 1 — flags PAA questions your content doesn't address
- Identifies SERP feature opportunities (e.g. "competitors have FAQ schema but you don't" → featured snippet opportunity)

**Gap detection logic:**
```python
from thefuzz import fuzz

def find_missing_topics(your_headings: list[str], 
                        competitor_headings: list[list[str]],
                        similarity_threshold: int = 70) -> list[str]:
    from collections import Counter
    
    # Count how many competitors cover each topic
    topic_counts = Counter()
    for headings in competitor_headings:
        for h in headings:
            topic_counts[h.lower().strip()] += 1
    
    your_topics = [h.lower().strip() for h in your_headings]
    
    missing = []
    for topic, count in topic_counts.items():
        if count < 2:
            continue
        # Check if any of your headings fuzzy-match this topic
        matched = any(
            fuzz.token_sort_ratio(topic, yh) >= similarity_threshold
            for yh in your_topics
        )
        if not matched:
            missing.append(topic)
    
    return missing[:20]
```

**Output:** List of missing topics/sections + SERP feature opportunities.

---

### Node 6 — `audit_internal_links`

**File:** `tools/internal_links.py`

**What it does:**
- Crawls the target page and checks internal linking structure
- Counts internal links pointing to the target page (from other pages on the same domain)
- Checks if the target page links to other relevant pages on the domain
- Calculates an internal link score (0.0 – 1.0) based on:
  - Number of internal links to/from the page
  - Anchor text relevance to the target keyword
  - Link depth from homepage
- Identifies issues: orphan pages, missing contextual links, poor anchor text

**Output:** Internal link score + list of specific internal linking issues.

---

### Node 7 — `synthesise_report`

**File:** `report/synthesiser.py`

**What it does:**
- Aggregates all tool outputs into a structured JSON per keyword
- Makes a single Anthropic API call per keyword with a detailed prompt
- Combines all per-keyword reports into one final document with cross-keyword summary

**Prompt structure:**
```python
REPORT_PROMPT = """
You are a senior SEO strategist. Analyse the data below and write a structured SEO report.

Domain: {target_domain}
Keyword: "{keyword}"

## Data

Current ranking: {your_rank} (URL: {your_url})

SERP Features present for this keyword:
{serp_features_formatted}

People Also Ask questions:
{paa_questions}

Top competitors:
{competitors_formatted}

On-page issues found on our page:
{on_page_issues}

Page Speed (Core Web Vitals):
{page_speed_formatted}

Topics our competitors cover that we are missing:
{missing_topics}

SERP feature opportunities:
{serp_feature_opportunities}

High-authority domains linking to competitors but not us:
{backlink_gaps}

Internal link score: {internal_link_score}/1.0
Internal link issues:
{internal_link_issues}

## Your Report Must Include

1. **Ranking summary** — where we stand and how far we are from page 1
2. **SERP landscape** — what SERP features exist and which ones we can target
3. **Why competitors outrank us** — specific, evidence-based reasons from the data
4. **Content gaps** — exactly what sections/topics we need to add, including PAA questions to answer
5. **Backlink opportunities** — top 5 domains to target and why
6. **On-page fixes** — prioritised list of technical fixes including page speed improvements
7. **Internal linking fixes** — specific pages to link to/from and anchor text suggestions
8. **Action plan** — split into:
   - Quick wins (can be done in under a day)
   - Medium-term (1–4 weeks)
   - Long-term (content + authority building)

Be specific and direct. Every suggestion must reference the data provided.
Do not be generic. If word count is the issue, say "add ~{word_gap} more words".
"""
```

---

## Caching Strategy (`cache/manager.py`)

All external API calls must be cached to avoid redundant requests and stay within rate limits.

```python
import hashlib
import json
import diskcache

cache = diskcache.Cache("./cache_store")

def cached_call(fn, *args, ttl_hours=24, **kwargs):
    key = hashlib.md5(
        json.dumps({"fn": fn.__name__, "args": args, "kwargs": kwargs}, 
                   sort_keys=True).encode()
    ).hexdigest()
    
    if key in cache:
        return cache[key]
    
    result = fn(*args, **kwargs)
    cache.set(key, result, expire=ttl_hours * 3600)
    return result
```

Cache all: SerpAPI calls, competitor page scrapes, Moz/Ahrefs API calls, PageSpeed API calls.
Do NOT cache: the LLM synthesis step (always regenerate the report).

---

## robots.txt Compliance (`tools/robots_check.py`)

Every scrape request must check `robots.txt` first.

```python
from urllib.robotparser import RobotFileParser
from functools import lru_cache

@lru_cache(maxsize=100)
def get_robot_parser(domain: str) -> RobotFileParser:
    rp = RobotFileParser()
    rp.set_url(f"{domain}/robots.txt")
    rp.read()
    return rp

def can_scrape(url: str, user_agent: str = "SEOBot") -> bool:
    domain = extract_base_domain(url)
    rp = get_robot_parser(domain)
    return rp.can_fetch(user_agent, url)
```

---

## Output Format

### JSON data file (`report_data.json`)
Full structured data for all keywords — useful for building dashboards later.

```json
{
  "domain": "https://yoursite.com",
  "generated_at": "2024-01-15T10:30:00Z",
  "keywords": [
    {
      "keyword": "example keyword",
      "your_rank": 14,
      "your_url": "https://yoursite.com/page",
      "serp_features": {
        "has_featured_snippet": true,
        "people_also_ask": ["What is...?", "How to...?"]
      },
      "top_competitors": [...],
      "on_page_issues": [...],
      "missing_topics": [...],
      "backlink_gap": [...],
      "internal_link_score": 0.6,
      "internal_link_issues": [...],
      "page_speed": {"lcp": 2.1, "cls": 0.05, "score": 78}
    }
  ]
}
```

### Markdown report (`seo_report.md`)
One section per keyword, written by the LLM. Final section is a cross-keyword summary with the top 10 highest-impact actions across all keywords.

---

## Entry Point (`main.py`)

```python
import asyncio
import argparse
from graph import build_graph
from config import settings

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, help="Target domain, e.g. https://yoursite.com")
    parser.add_argument("--keywords", required=True, nargs="+", help="Keywords to analyse")
    parser.add_argument("--output", default="./output", help="Output directory")
    args = parser.parse_args()
    
    graph = build_graph()
    
    initial_state = {
        "target_domain": args.domain,
        "keywords": args.keywords,
        "results": [],
        "final_report": None,
        "errors": []
    }
    
    final_state = await graph.ainvoke(initial_state)
    
    # Write outputs
    write_json(final_state["results"], args.output)
    write_markdown(final_state["final_report"], args.output)
    
    print(f"Report saved to {args.output}/")

if __name__ == "__main__":
    asyncio.run(main())
```

**Usage:**
```bash
python main.py \
  --domain https://yoursite.com \
  --keywords "seo tools" "best seo software" "keyword research tool" \
  --output ./reports
```

---

## Error Handling

- Each tool node must catch exceptions per keyword individually — a failure on one keyword must not stop the pipeline for other keywords
- Log all errors to `state["errors"]` as strings: `"[check_rankings] keyword='X' error: timeout"`
- The synthesis node must handle partial data gracefully — if backlink data is missing, skip that section rather than failing
- Wrap all `httpx` scraping calls in retry logic with exponential backoff (3 retries, start 2s)

---

## Requirements (`requirements.txt`)

```
langchain>=0.2.0
langgraph>=0.1.0
anthropic>=0.28.0
langchain-anthropic>=0.3.0
httpx>=0.27.0
beautifulsoup4>=4.12.0
requests>=2.32.0
diskcache>=5.6.0
pydantic-settings>=2.0.0
thefuzz>=0.22.0
python-levenshtein>=0.25.0
python-dotenv>=1.0.0
```

---

## Implementation Order (Step by Step)

Build and test in this order — each step is independently testable:

1. **`state.py` + `config.py`** — schema and settings
2. **`cache/manager.py`** — caching layer (test with mock data)
3. **`tools/robots_check.py`** — robots.txt checker (test against known sites)
4. **`tools/serp.py`** — SERP checker + PAA + features (test with 1 keyword)
5. **`tools/competitor.py`** — competitor scraper (test on a known URL)
6. **`tools/onpage.py`** — on-page auditor (test against your own site)
7. **`tools/pagespeed.py`** — PageSpeed integration (test with 1 URL)
8. **`tools/backlinks.py`** — backlink gap tool (test Moz/Ahrefs connection)
9. **`tools/content_gap.py`** — gap detection with thefuzz (test with mock data)
10. **`tools/internal_links.py`** — internal link auditor (test on your own site)
11. **`graph.py`** — wire all nodes into LangGraph pipeline
12. **`report/synthesiser.py`** — test report generation with pre-collected data
13. **`main.py` + `report/formatter.py`** — full end-to-end run

---

## Notes and Constraints

- Respect `robots.txt` on competitor scraping — enforced via `tools/robots_check.py`
- SerpAPI free tier: 100 searches/month. For production use, budget ~$50/month for the basic plan
- Moz free tier: 10 requests/10 seconds. For more volume, use Ahrefs ($99/month) or SEMrush API
- Do not run more than 3 concurrent scraping requests to avoid IP bans (enforced via asyncio.Semaphore)
- The agent is keyword-level, not page-level — it analyses the best-ranking page for each keyword, not the entire site
- For domains not ranking in top 100, the agent should still run the competitor and content gap analysis using the homepage or the most relevant page (passed manually or auto-detected)
- Google PageSpeed Insights API is free (25k queries/day) — no billing required, just an API key from Google Cloud Console
