# Postsyntax Content Pipeline

> Business logic documentation for the automated content creation system.

## Executive Summary

The Postsyntax pipeline automates the content creation lifecycle from **discovery to publication**. It monitors competitor publications, identifies high-performing content, generates original topic ideas, and produces publication-ready drafts—all while maintaining your unique voice and avoiding duplicate content.

**Core Value Proposition:**
- Find what's working in your niche (data-driven discovery)
- Generate ideas that tap into proven demand
- Produce drafts faster while maintaining quality
- Never accidentally duplicate existing content

---

## The Pipeline at a Glance

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONTENT CREATION PIPELINE                            │
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │  DISCOVER    │    │   INGEST     │    │   SELECT     │    │  PRODUCE  │ │
│  │              │    │              │    │              │    │           │ │
│  │ Find top     │───▶│ Sync posts   │───▶│ Queue best   │───▶│ Draft     │ │
│  │ publications │    │ + embeddings │    │ articles     │    │ articles  │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│         │                                       │                           │
│         │            ┌──────────────┐           │                           │
│         └───────────▶│  GENERATE    │◀──────────┘                           │
│                      │              │                                        │
│                      │ Create new   │                                        │
│                      │ topic ideas  │                                        │
│                      └──────────────┘                                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Workflows

### 1. Publication Discovery

**Purpose:** Find high-quality source publications in your niche.

**When to use:** When entering a new topic area or refreshing your source list.

**What it does:**
1. Scrapes Substack leaderboards (top 75 paid + 75 trending)
2. Fetches recent and top posts from each publication
3. Scores every post for keyword relevance using AI
4. Ranks publications using Borda scoring (4 signals combined)
5. Produces a ranked list of publications to follow

**Inputs:**
| Field | Description |
|-------|-------------|
| Category | Substack category (technology, business, etc.) |
| Keywords | Topics to match (e.g., "System Design", "Distributed Systems") |
| Top N | How many publications to return |

**Outputs:**
- Ranked list of publication handles
- Stored in KV for downstream workflows
- All discovered posts saved to database (crash-safe)

**Business Logic:**
- Borda ranking combines: leaderboard type, leaderboard rank, LLM relevance scores, and engagement metrics
- Posts are stored immediately as fetched (no data loss on failure)
- LLM scoring is cached to avoid redundant API calls

---

### 2. Content Sync

**Purpose:** Keep your database up-to-date with source publications.

**When to use:** Run regularly (daily/weekly) to ingest new content.

**What it does:**
1. Fetches post listings from source publications
2. Identifies posts not yet in database
3. Scrapes full content and stores immediately
4. Generates vector embeddings for semantic search

**Inputs:**
| Field | Description |
|-------|-------------|
| Source Handles | Publication handles to sync (e.g., "designgurus, stratechery") |

**Outputs:**
- New posts stored with full content
- Vector embeddings for semantic deduplication
- Metrics (likes, comments) captured

**Business Logic:**
- Crash-safe: each post stored immediately (no batch failures)
- Retry logic: 3 attempts with 30s delay on failures
- Supports paid content via Substack session cookies
- Embeddings enable semantic similarity searches

---

### 3. Queue Source Articles

**Purpose:** Select high-performing articles to inspire or adapt.

**When to use:** When you want to queue proven content for drafting.

**What it does:**
1. Queries the posts database with filters (min likes, max age)
2. Excludes articles already in queue or published
3. Picks a mix of recent and top performers
4. Adds selected articles to production queue

**Inputs:**
| Field | Description |
|-------|-------------|
| Source Handles | Publications to draw from |
| Target Handle | Your publication (for queue scoping) |
| Count | Number of articles to queue |
| Min Likes | Engagement threshold |
| Max Age Days | Recency filter |

**Outputs:**
- Articles added to production queue with `status: pending`
- Ready for Factory workflow

**Business Logic:**
- Alternates between recent (last 7 days) and top performers for variety
- Only queues articles with full content available
- Prevents duplicate queueing

---

### 4. Generate Ideas

**Purpose:** Create original topic ideas based on proven demand.

**When to use:** When you want fresh ideas, not adaptations of existing articles.

**What it does:**
1. Fetches top posts from source publications
2. Filters out posts already used for ideas
3. Loads your published posts for voice/style context
4. AI generates ideas combining market demand + your voice
5. Deduplicates against existing queue and published content
6. Queues unique ideas for production

**Inputs:**
| Field | Description |
|-------|-------------|
| Target Handle | Your publication |
| Publications | Source handles (or KV key from discovery) |
| Idea Count | How many ideas to generate |
| Posts Per Source | Top posts to analyze per publication |

**Outputs:**
- Original topic ideas in production queue
- Each idea includes: title, summary, post type (free/paid), inspiration notes
- Linked to source posts that inspired them

**Business Logic:**
- Ideas are **original topics**, not adaptations—inspired by trends but unique
- Generates both paid (premium deep-dives) and free (growth) content suggestions
- Semantic deduplication prevents overlapping with existing content
- Your voice/style is learned from your last 10 published posts

---

### 5. The Factory

**Purpose:** Transform queued items into publication-ready drafts.

**When to use:** When you have items in queue ready to draft.

**What it does:**
1. Fetches next pending item (or specific ID)
2. Locks it to prevent concurrent drafting
3. Loads source context (if adapting an article)
4. AI writes the full article
5. AI inserts image placeholders and paywall marker
6. Generates image prompts for each placeholder
7. Optionally generates actual images
8. Saves draft for review

**Inputs:**
| Field | Description |
|-------|-------------|
| Target Handle | Your publication |
| Queue ID | (Optional) Specific item to draft |
| Length Mode | Brief (~700), Standard (~1250), Long (~2500), or Custom |
| Article Model | AI model for writing |
| Min/Max Images | Image count constraints |
| Generate Images | Whether to create actual images |
| Image Model | AI model for images (if enabled) |

**Outputs:**
- Full article with `<image:1>`, `<image:2>`, `<paywall>` placeholders
- Image prompts keyed by placeholder
- Generated images (if enabled)
- Draft metadata (model used, word count, etc.)

**Business Logic:**
- Crash recovery: stale items (stuck >30 min) auto-reset on startup
- Error recovery: failed drafts reset to pending for retry
- Placeholders allow manual image generation/review
- Supports multiple AI models for writing and images

---

## Workflow Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   publication_discovery ──────┐                                  │
│   (find sources)              │                                  │
│                               ▼                                  │
│                         ┌─────────────┐                         │
│                         │  KV Store   │                         │
│                         │ (sources)   │                         │
│                         └─────────────┘                         │
│                               │                                  │
│           ┌───────────────────┴───────────────────┐             │
│           ▼                                       ▼             │
│   ┌───────────────┐                     ┌─────────────────┐     │
│   │ content_sync  │                     │ generate_ideas  │     │
│   │ (ingest)      │                     │ (create topics) │     │
│   └───────────────┘                     └─────────────────┘     │
│           │                                       │             │
│           ▼                                       │             │
│   ┌───────────────┐                               │             │
│   │ queue_sources │                               │             │
│   │ (select)      │                               │             │
│   └───────────────┘                               │             │
│           │                                       │             │
│           └───────────────────┬───────────────────┘             │
│                               ▼                                  │
│                       ┌─────────────┐                           │
│                       │  QUEUE      │                           │
│                       │ (pending)   │                           │
│                       └─────────────┘                           │
│                               │                                  │
│                               ▼                                  │
│                       ┌─────────────┐                           │
│                       │  factory    │                           │
│                       │ (produce)   │                           │
│                       └─────────────┘                           │
│                               │                                  │
│                               ▼                                  │
│                       ┌─────────────┐                           │
│                       │   DRAFT     │                           │
│                       │ (review)    │                           │
│                       └─────────────┘                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Two paths to the queue:**
1. **Adaptation path:** content_sync → queue_sources → factory
2. **Ideation path:** generate_ideas → factory

Both paths can use discovery results to find sources.

---

## Data Flow

### Key Tables

| Table | Purpose |
|-------|---------|
| `posts` | All scraped posts from source publications |
| `production_queue` | Items being drafted (pending → drafting → review → published) |
| `target_items` | Your published content (for dedup and voice) |
| `idea_source_posts` | Links ideas to their inspiration sources |

### Content Lifecycle

```
Source Post (external)
    │
    ▼
posts table (scraped)
    │
    ├──▶ queue_sources ──▶ production_queue (as adaptation)
    │
    └──▶ generate_ideas ──▶ production_queue (as original idea)
                                    │
                                    ▼
                              factory drafts
                                    │
                                    ▼
                              production_queue (status: review)
                                    │
                                    ▼ (manual publish)
                              target_items (published)
```

---

## Typical Usage Patterns

### Starting Fresh (New Niche)

1. Run **Publication Discovery** with your category + keywords
2. Review ranked publications, curate your source list
3. Run **Content Sync** to ingest their content
4. Run **Generate Ideas** to create original topics
5. Run **Factory** to draft your first articles

### Ongoing Content Production

1. **Weekly:** Run Content Sync to stay updated
2. **As needed:** Run Queue Sources to add proven articles
3. **As needed:** Run Generate Ideas for fresh topics
4. **Daily:** Run Factory to produce drafts from queue

### Research Mode

1. Run **Publication Discovery** with different keywords
2. Compare results across categories
3. Identify emerging publications before they peak

---

## Version History

| Workflow | Version | Key Changes |
|----------|---------|-------------|
| content_sync_v3 | 3.3 | Crash-safe storage, retry logic |
| queue_sources_v3 | 3.0 | Unified posts table, direct handle input |
| generate_ideas | 3.0 | KV integration, voice learning |
| factory | 5.4 | Image injection, multiple AI models |
| publication_discovery | 4.0 | Borda ranking, crash-safe posts |
