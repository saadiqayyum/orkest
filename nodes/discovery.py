"""
Discovery node functions for finding Substack publications.

These workflow nodes handle publication discovery:
- scrape_leaderboards: Fetch top publications from Substack leaderboard
- fetch_publication_posts: Get recent and top posts from each publication
- score_keyword_relevance: Score by keyword frequency (fast filter)
- score_llm_relevance: Score by LLM relevance (accurate ranking)
- generate_discovery_report: Create final report

DEV NOTE: Always use verbose ctx.report_input/report_output calls.
Include all relevant fields, samples of data, counts, and debugging info.
This is a workflow standard for dg-team - users expect detailed visibility.

NOTE: This module uses inline fetching for discovery-specific needs (recent + top posts
separately for scoring). For general post fetching, use shared.substack_client.SubstackClient.
"""
import os
import re
import json
import asyncio
import structlog
import httpx
from typing import Optional, List, Dict, Any
from datetime import datetime

from .schemas import (
    LLMConfig, resolve_model,
    ScrapeLeaderboardsInput, ScrapeLeaderboardsOutput, DiscoveredPublication,
    FetchPublicationPostsInput, FetchPublicationPostsOutput, DiscoveredPost,
    ScoreKeywordRelevanceInput, ScoreKeywordRelevanceOutput,
    ScoreLLMRelevanceInput, ScoreLLMRelevanceOutput,
    GenerateDiscoveryReportInput, GenerateDiscoveryReportOutput,
    # Report generation (uses ScoredPublicationResult from rank_publications_llm)
    ScoredPublicationResult, RankPublicationsOutput,
    GenerateReportInput, GenerateReportOutput,
    # LLM-based scoring schemas
    ScorePostsLLMInput, ScorePostsLLMOutput, ScoredPublication, PostScore,
    RankPublicationsLLMInput, RankPublicationsLLMOutput, RankedPublication,
    # Unified posts schemas
    PostData, UpsertPostsInput, UpsertPostsOutput,
)
from shared.substack_client import SubstackClient, SubstackPost, create_client

logger = structlog.get_logger()

# Rate limiting
REQUEST_DELAY = 0.2  # 200ms between requests
CONCURRENT_WORKERS = 5


def _substack_post_to_discovered(post: SubstackPost, post_type: str) -> DiscoveredPost:
    """Convert SubstackPost (from client) to DiscoveredPost (for discovery workflow)."""
    # Merge subtitle and description - keep whichever is longer (or present)
    subtitle = post.subtitle or ""
    description = post.description or ""
    merged_subtitle = subtitle if len(subtitle) >= len(description) else description

    return DiscoveredPost(
        publication_handle=post.publication_handle,
        url=post.url,
        title=post.title,
        subtitle=merged_subtitle,
        content_raw=post.content_raw,
        author=post.author,
        substack_post_id=post.substack_post_id,
        slug=post.slug,
        tags=post.tags if post.tags else [],
        section_name=post.section_name or "",
        post_type=post_type,  # 'recent' or 'top' (discovery context)
        content_type=post.post_type or "",  # 'newsletter', 'podcast', 'thread'
        audience=post.audience or "",
        published_at=post.published_at.isoformat() if post.published_at else None,
        likes_count=post.likes_count,
        comments_count=post.comments_count,
        word_count=post.word_count or 0,
    )


def _get_param(params, key, default=None):
    """Get param from Pydantic model. Params should always be Pydantic - runner converts dicts."""
    return getattr(params, key, default)


def _ensure_publication(pub) -> DiscoveredPublication:
    """Convert dict to DiscoveredPublication if needed."""
    if isinstance(pub, DiscoveredPublication):
        return pub
    if isinstance(pub, dict):
        # Convert nested posts
        posts = pub.get("posts") or []
        pub["posts"] = [_ensure_post(p) for p in posts]
        return DiscoveredPublication(**pub)
    return pub


def _ensure_post(post) -> DiscoveredPost:
    """Convert dict to DiscoveredPost if needed."""
    if isinstance(post, DiscoveredPost):
        return post
    if isinstance(post, dict):
        return DiscoveredPost(**post)
    return post


def _ensure_publications(publications: List) -> List[DiscoveredPublication]:
    """Convert list of dicts to DiscoveredPublication models."""
    return [_ensure_publication(p) for p in (publications or [])]


async def scrape_leaderboards(
    ctx,
    params: ScrapeLeaderboardsInput,
) -> ScrapeLeaderboardsOutput:
    """
    Fetch top publications from Substack leaderboard for a category.
    Gets top 100 paid + top 100 trending, deduplicates.
    """
    import json
    category = _get_param(params, "category", "").lower()

    ctx.report_input({
        "category": category,
        "target_leaderboard_types": ["paid", "trending"],
        "target_per_type": 100,
        "expected_publications": "~100-200 (100 paid + 100 trending, deduplicated)",
    })

    try:
        # Use direct HTTP client for leaderboard APIs (no caching - data changes frequently)
        async with httpx.AsyncClient(
            timeout=30,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SubstackAnalyzer/1.0)"}
        ) as client:

            # Step 1: Get category ID
            categories_url = "https://substack.com/api/v1/categories?purpose=leaderboard"
            resp = await client.get(categories_url)
            categories = resp.json()

            category_id = None
            category_name = None
            for cat in categories:
                if cat.get("name", "").lower() == category or cat.get("slug", "").lower() == category:
                    category_id = cat["id"]
                    category_name = cat.get("name", category)
                    break

            if not category_id:
                available = [c["name"] for c in categories]
                ctx.report_output({
                    "status": "error",
                    "error": f"Category '{category}' not found",
                    "available_categories": available,
                })
                return ScrapeLeaderboardsOutput(status="error")

            logger.info("category_found", category=category, category_id=category_id)
            ctx.report_progress(10, f"Found category '{category_name}', fetching leaderboards...")

            # Step 2: Fetch leaderboard pages (no caching - leaderboard data changes)
            publications = []
            seen_handles = set()
            fetch_stats = {"paid": {"pages": 0, "items": 0}, "trending": {"pages": 0, "items": 0}}
            target_per_type = 100  # Stop after 100 publications per type
            max_pages = 10  # Safety limit
            pages_fetched = 0

            for leaderboard_type in ["paid", "trending"]:
                for page in range(max_pages):
                    # Stop if we have enough for this type
                    if fetch_stats[leaderboard_type]["items"] >= target_per_type:
                        break
                    url = f"https://substack.com/api/v1/category/leaderboard/{category_id}/{leaderboard_type}?page={page}"

                    try:
                        resp = await client.get(url)
                        data = resp.json()

                        # Response structure is: { items: [ { publication: {...} }, ... ] }
                        items = data.get("items", [])

                        if not items:
                            logger.info("no_more_items",
                                       type=leaderboard_type,
                                       page=page)
                            break  # No more pages

                        fetch_stats[leaderboard_type]["pages"] += 1
                        fetch_stats[leaderboard_type]["items"] += len(items)

                        # Debug: log first item structure on first page
                        if page == 0 and items:
                            first_item = items[0]
                            pub_sample = first_item.get("publication", {})
                            logger.info("leaderboard_item_structure",
                                       type=leaderboard_type,
                                       item_keys=list(first_item.keys()),
                                       pub_keys=list(pub_sample.keys()) if pub_sample else None,
                                       sample_name=pub_sample.get("name"),
                                       sample_subdomain=pub_sample.get("subdomain"))

                        for idx, item in enumerate(items):
                            # Publication is always nested under item.publication
                            pub = item.get("publication")
                            if not pub or not isinstance(pub, dict):
                                continue

                            handle = pub.get("subdomain", "")
                            if not handle or handle in seen_handles:
                                continue

                            seen_handles.add(handle)

                            # Get subscriber count from various fields
                            subscriber_count = (
                                pub.get("rankingDetailFreeSubscriberCount") or
                                pub.get("freeSubscriberCount") or
                                pub.get("subscriber_count") or
                                0
                            )

                            publications.append(DiscoveredPublication(
                                handle=handle,
                                name=pub.get("name", handle),
                                description=pub.get("description") or pub.get("hero_text") or "",
                                subscriber_count=subscriber_count,
                                custom_domain=pub.get("custom_domain"),
                                author_name=pub.get("author_name", ""),
                                category=category,
                                leaderboard_type=leaderboard_type,
                                leaderboard_rank=page * 25 + idx + 1,
                            ))

                        logger.info("leaderboard_page_fetched",
                                   type=leaderboard_type, page=page,
                                   count=len(items), total=len(publications))

                    except Exception as e:
                        logger.error("leaderboard_page_error", type=leaderboard_type, page=page, error=str(e))
                        # Fail loudly on first network failure - don't continue with partial data
                        raise RuntimeError(f"Failed to fetch {leaderboard_type} leaderboard page {page}: {e}") from e

                    pages_fetched += 1
                    # Progress: estimate based on target (100 per type = 200 total)
                    total_items = fetch_stats["paid"]["items"] + fetch_stats["trending"]["items"]
                    pct = int(10 + min(80, total_items / 2))  # 10-90%
                    ctx.report_progress(pct, f"Fetched {leaderboard_type} page {page + 1} ({len(publications)} publications)")

                    await asyncio.sleep(REQUEST_DELAY)

        ctx.report_progress(95, f"Done! Found {len(publications)} unique publications")

        ctx.report_output({
            "status": "success",
            "category_id": category_id,
            "category_name": category_name,
            "publications_count": len(publications),
            "unique_handles": len(seen_handles),
            "fetch_stats": fetch_stats,
            "leaderboard_breakdown": {
                "paid": len([p for p in publications if p.leaderboard_type == "paid"]),
                "trending": len([p for p in publications if p.leaderboard_type == "trending"]),
            },
            # Show ALL publications - this is important data
            "publications": [
                {
                    "handle": p.handle,
                    "name": p.name,
                    "subscribers": p.subscriber_count,
                    "type": p.leaderboard_type,
                    "rank": p.leaderboard_rank,
                }
                for p in publications
            ],
            "deduplication_note": f"Started with {fetch_stats['paid']['items'] + fetch_stats['trending']['items']} items, deduplicated to {len(publications)} unique publications",
        })

        return ScrapeLeaderboardsOutput(
            publications=publications,
            total_count=len(publications),
            status="success",
        )

    except Exception as e:
        logger.error("scrape_leaderboards_failed", error=str(e))
        ctx.report_output({
            "status": "error",
            "error": str(e),
            "category": category,
        })
        return ScrapeLeaderboardsOutput(status="error")


async def fetch_publication_posts(
    ctx,
    params: FetchPublicationPostsInput,
) -> FetchPublicationPostsOutput:
    """
    Fetch recent and top posts for each discovered publication.
    Uses concurrent workers for speed.
    """
    publications = _ensure_publications(_get_param(params, "publications"))
    recent_count = _get_param(params, "recent_count") or 3
    top_count = _get_param(params, "top_count") or 3

    ctx.report_input({
        "publications_count": len(publications),
        "recent_count": recent_count,
        "top_count": top_count,
        "max_posts_per_publication": recent_count + top_count,
        "concurrent_workers": CONCURRENT_WORKERS,
        "publications": [p.handle for p in publications],
    })

    # Track errors for reporting
    fetch_errors = []

    # Create shared SubstackClient for all fetches
    # Note: Caching is now controlled per-method - fetch_post_index has no cache
    client = await create_client()

    async def fetch_posts_for_publication(pub: DiscoveredPublication) -> List[DiscoveredPost]:
        """Fetch posts for a single publication using SubstackClient."""
        posts = []
        recent_error = None
        top_error = None
        seen_urls = set()

        # Fetch recent posts using SubstackClient
        try:
            recent_posts = await client.fetch_post_index(pub.handle, sort="new", limit=recent_count)
            for sp in recent_posts:
                if sp.url not in seen_urls:
                    seen_urls.add(sp.url)
                    posts.append(_substack_post_to_discovered(sp, "recent"))
        except Exception as e:
            recent_error = str(e)
            logger.error("fetch_recent_posts_error", handle=pub.handle, error=str(e))

        await asyncio.sleep(REQUEST_DELAY)

        # Fetch top posts using SubstackClient
        try:
            top_posts = await client.fetch_post_index(pub.handle, sort="top", limit=top_count)
            for sp in top_posts:
                # Skip if already added from recent
                if sp.url not in seen_urls:
                    seen_urls.add(sp.url)
                    posts.append(_substack_post_to_discovered(sp, "top"))
        except Exception as e:
            top_error = str(e)
            logger.error("fetch_top_posts_error", handle=pub.handle, error=str(e))

        # Track errors for final report
        if recent_error or top_error:
            fetch_errors.append({
                "handle": pub.handle,
                "recent_error": recent_error,
                "top_error": top_error,
                "posts_fetched": len(posts),
            })

        return posts

    # Process publications with concurrent workers
    all_posts = []
    completed = 0
    failed = 0
    total = len(publications)
    total_likes = 0

    # Helper to convert DiscoveredPost to PostData for storage
    def to_post_data(post: DiscoveredPost) -> PostData:
        return PostData(
            url=post.url,
            publication_handle=post.publication_handle,
            title=post.title,
            subtitle=post.subtitle or None,  # Already merged in _substack_post_to_discovered
            content_raw=post.content_raw,
            author=post.author,
            substack_post_id=post.substack_post_id,
            slug=post.slug,
            post_type=post.content_type or None,  # 'newsletter', 'podcast', 'thread'
            audience=post.audience or None,
            word_count=post.word_count if post.word_count > 0 else None,
            tags=post.tags if hasattr(post, 'tags') and post.tags else [],
            section_name=post.section_name if hasattr(post, 'section_name') else None,
            is_paywalled=post.audience in ("only_paid", "founding"),
            has_full_content=bool(post.content_raw),
            likes_count=post.likes_count,
            comments_count=post.comments_count,
            shares_count=0,
            published_at=post.published_at,
            discovery_source="publication_discovery",
        )

    # Import upsert function for storing posts
    from .db_ops import upsert_posts as _upsert_posts

    # Track storage stats
    total_inserted = 0
    total_updated = 0

    async def worker(queue: asyncio.Queue):
        nonlocal completed, failed, total_likes, total_inserted, total_updated
        while True:
            pub = None
            try:
                pub = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                break

            try:
                posts = await fetch_posts_for_publication(pub)
                # Attach posts to publication
                pub.posts = posts
                all_posts.extend(posts)

                # Store posts immediately after fetching (crash-safe)
                if posts:
                    try:
                        post_data_list = [to_post_data(p) for p in posts]
                        upsert_input = UpsertPostsInput(
                            posts=post_data_list,
                            discovery_source="publication_discovery",
                        )
                        result = await _upsert_posts(ctx, upsert_input)
                        total_inserted += result.inserted_count
                        total_updated += result.updated_count
                    except Exception as e:
                        logger.error("store_posts_error", handle=pub.handle, error=str(e))

                completed += 1
                if not posts:
                    failed += 1

                # Track engagement
                pub_likes = sum(p.likes_count for p in posts)
                total_likes += pub_likes

                # Report progress frequently so user sees activity
                if completed % 5 == 0 or completed == total:
                    ctx.report_progress(
                        int(completed / total * 100),
                        f"Fetched {completed}/{total} pubs ({len(all_posts)} posts, {total_inserted} new, {total_updated} updated)"
                    )
            except Exception as e:
                # Catch any unhandled exception to prevent queue.join() from hanging
                logger.error("worker_error", handle=pub.handle if pub else "unknown", error=str(e))
                failed += 1
                completed += 1
            finally:
                queue.task_done()

    # Create queue and start workers
    queue = asyncio.Queue()
    for pub in publications:
        await queue.put(pub)

    workers = [asyncio.create_task(worker(queue)) for _ in range(CONCURRENT_WORKERS)]
    await queue.join()

    for w in workers:
        w.cancel()

    # Compute stats
    pubs_with_posts = [p for p in publications if p.posts]
    pubs_without_posts = [p for p in publications if not p.posts]
    posts_by_type = {"recent": 0, "top": 0}
    posts_with_content = 0
    for post in all_posts:
        posts_by_type[post.post_type] = posts_by_type.get(post.post_type, 0) + 1
        if post.content_raw:
            posts_with_content += 1

    # Calculate engagement stats
    total_likes = sum(p.likes_count for p in all_posts)
    top_posts_by_likes = sorted(all_posts, key=lambda p: p.likes_count, reverse=True)[:10]

    # Field coverage stats - show ACTUAL data quality for all fields
    field_coverage = {
        "total_posts": len(all_posts),
        "with_title": sum(1 for p in all_posts if p.title),
        "with_subtitle": sum(1 for p in all_posts if p.subtitle),
        "with_tags": sum(1 for p in all_posts if hasattr(p, 'tags') and p.tags),
        "with_section_name": sum(1 for p in all_posts if hasattr(p, 'section_name') and p.section_name),
        "with_author": sum(1 for p in all_posts if p.author and p.author != "Unknown"),
        "with_content_raw": posts_with_content,
    }

    # Find a sample post WITH tags to show what good data looks like
    sample_post_with_tags = next((p for p in all_posts if hasattr(p, 'tags') and p.tags), None)
    sample_with_tags = None
    if sample_post_with_tags:
        sample_with_tags = {
            "title": sample_post_with_tags.title,
            "tags": sample_post_with_tags.tags,
            "section_name": sample_post_with_tags.section_name if hasattr(sample_post_with_tags, 'section_name') else "",
            "subtitle": sample_post_with_tags.subtitle[:150] + "..." if sample_post_with_tags.subtitle and len(sample_post_with_tags.subtitle) > 150 else sample_post_with_tags.subtitle,
            "publication_handle": sample_post_with_tags.publication_handle,
        }

    # Collect unique tags across all posts
    all_tags = []
    for p in all_posts:
        if hasattr(p, 'tags') and p.tags:
            all_tags.extend(p.tags)
    unique_tags = list(set(all_tags))[:20]  # Top 20 unique tags

    # Cleanup client
    await client.close()

    ctx.report_output({
        "status": "success",
        "publications_processed": len(publications),
        "publications_with_posts": len(pubs_with_posts),
        "publications_without_posts": len(pubs_without_posts),
        "total_posts": len(all_posts),
        "posts_by_type": posts_by_type,
        "avg_posts_per_publication": round(len(all_posts) / len(publications), 1) if publications else 0,
        "storage": {
            "posts_inserted": total_inserted,
            "posts_updated": total_updated,
            "total_stored": total_inserted + total_updated,
        },
        "field_coverage": field_coverage,
        "sample_post_with_tags": sample_with_tags,
        "unique_tags_found": unique_tags,
        "engagement_summary": {
            "total_likes": total_likes,
            "avg_likes_per_post": round(total_likes / len(all_posts), 1) if all_posts else 0,
        },
        "top_posts_by_likes": [
            {
                "title": p.title,
                "subtitle": p.subtitle[:200] + "..." if p.subtitle and len(p.subtitle) > 200 else p.subtitle if p.subtitle else None,
                "tags": p.tags if hasattr(p, 'tags') and p.tags else None,
                "section_name": p.section_name if hasattr(p, 'section_name') and p.section_name else None,
                "publication_handle": p.publication_handle,
                "likes_count": p.likes_count,
                "comments_count": p.comments_count,
                "word_count": p.word_count if p.word_count else None,
                "audience": p.audience if p.audience else None,
            }
            for p in top_posts_by_likes[:5]  # Top 5 with all fields
        ],
        "errors": {
            "total_failed": len(pubs_without_posts),
            "failed_handles": [p.handle for p in pubs_without_posts],
            "partial_failures": len([e for e in fetch_errors if e["posts_fetched"] > 0]),
            "error_details": fetch_errors,
        },
    })

    return FetchPublicationPostsOutput(
        publications=publications,
        total_posts=len(all_posts),
        status="success",
    )


async def score_keyword_relevance(
    ctx,
    params: ScoreKeywordRelevanceInput,
) -> ScoreKeywordRelevanceOutput:
    """
    Score publications by keyword frequency in their posts.
    Fast first-pass filter to reduce candidates for LLM scoring.

    Expands keywords to include common abbreviations and related terms.
    """
    publications = _ensure_publications(_get_param(params, "publications"))
    keywords_str = _get_param(params, "keywords") or ""
    top_n = _get_param(params, "top_n") or 50

    # Parse keywords (semicolon-separated)
    raw_keywords = [k.strip().lower() for k in keywords_str.split(";") if k.strip()]

    # Keyword expansion map - maps phrases to related terms/abbreviations
    KEYWORD_EXPANSIONS = {
        "artificial intelligence": [
            "artificial intelligence", "ai", "machine learning", "deep learning",
            "neural network", "llm", "large language model", "gpt", "chatgpt",
            "ml", "data science", "computer vision", "nlp", "natural language",
            "transformer", "generative ai", "ai research", "openai", "anthropic",
            "claude", "gemini", "copilot",
        ],
        "machine learning": [
            "machine learning", "ml", "deep learning", "neural network",
            "training", "model", "inference", "tensorflow", "pytorch",
        ],
        "system design": [
            "system design", "distributed systems", "software architecture",
            "scalability", "microservices", "backend", "database design",
            "software engineering", "infrastructure",
        ],
        "prompt engineering": [
            "prompt", "prompting", "prompt engineering", "llm", "gpt",
            "chatgpt", "ai", "few-shot", "chain of thought",
        ],
        "technology": [
            "tech", "software", "coding", "developer", "programming",
            "engineering", "startup", "silicon valley",
        ],
        "crypto": [
            "crypto", "cryptocurrency", "bitcoin", "ethereum", "blockchain",
            "web3", "defi", "nft",
        ],
        "finance": [
            "finance", "investing", "stocks", "markets", "trading",
            "portfolio", "wealth", "money",
        ],
    }

    # Expand keywords
    expanded_keywords = set()
    for kw in raw_keywords:
        # Add the original keyword
        expanded_keywords.add(kw)
        # Add individual words from multi-word keywords
        for word in kw.split():
            if len(word) > 2:  # Skip tiny words
                expanded_keywords.add(word)
        # Add expansions if available
        if kw in KEYWORD_EXPANSIONS:
            expanded_keywords.update(KEYWORD_EXPANSIONS[kw])

    keywords = list(expanded_keywords)

    ctx.report_input({
        "publications_count": len(publications),
        "raw_keywords": raw_keywords,
        "expanded_keywords": keywords,
        "expansion_count": len(keywords),
        "top_n": top_n,
        "total_posts_to_score": sum(len(p.posts) for p in publications),
        "publications": [{"handle": p.handle, "name": p.name, "posts_count": len(p.posts)} for p in publications],
    })

    def score_publication(pub: DiscoveredPublication) -> float:
        """Calculate keyword relevance score for a publication."""
        score = 0.0

        # Score from publication metadata
        name_lower = pub.name.lower()
        desc_lower = (pub.description or "").lower()

        for keyword in keywords:
            # Title/name matches weighted higher
            if keyword in name_lower:
                score += 10
            if keyword in desc_lower:
                score += 3

        # Score from posts
        for post in pub.posts:
            title_lower = post.title.lower()
            subtitle_lower = (post.subtitle or "").lower()

            for keyword in keywords:
                # Title matches weighted very high
                if keyword in title_lower:
                    score += 5
                if keyword in subtitle_lower:
                    score += 2

        # Normalize by number of posts to avoid bias toward publications with more posts
        if pub.posts:
            score = score / len(pub.posts)

        return score

    # Score all publications
    for pub in publications:
        pub.keyword_score = score_publication(pub)

    # Sort by score and take top N
    publications.sort(key=lambda p: p.keyword_score or 0, reverse=True)
    filtered = publications[:top_n]

    # Check if any matches
    if not filtered or (filtered[0].keyword_score or 0) == 0:
        ctx.report_output({
            "status": "no_matches",
            "message": "No publications matched the keywords",
        })
        return ScoreKeywordRelevanceOutput(
            filtered_publications=filtered,
            status="no_matches",
        )

    # Score distribution
    all_scores = [p.keyword_score or 0 for p in publications]
    zero_score_count = sum(1 for s in all_scores if s == 0)

    ctx.report_output({
        "status": "success",
        "total_scored": len(publications),
        "filtered_count": len(filtered),
        "zero_score_count": zero_score_count,
        "score_distribution": {
            "min": round(min(all_scores), 2) if all_scores else 0,
            "max": round(max(all_scores), 2) if all_scores else 0,
            "avg": round(sum(all_scores) / len(all_scores), 2) if all_scores else 0,
        },
        "filtered_publications": [
            {"handle": p.handle, "name": p.name, "keyword_score": round(p.keyword_score or 0, 2)}
            for p in filtered
        ],
        "excluded_publications": [
            {"handle": p.handle, "name": p.name, "keyword_score": round(p.keyword_score or 0, 2)}
            for p in publications[top_n:]
        ],
        "keywords_matched": keywords,
    })

    return ScoreKeywordRelevanceOutput(
        filtered_publications=filtered,
        status="success",
    )


async def score_llm_relevance(
    ctx,
    params: ScoreLLMRelevanceInput,
) -> ScoreLLMRelevanceOutput:
    """
    Use LLM to score publications for relevance to keywords.

    Strategy:
    1. Batch 10 publications per LLM call (with all their post titles)
    2. Run batches in parallel for speed
    3. Filter out irrelevant (score < 40)
    4. From relevant pool: pick N/2 most relevant + N/2 most popular (by likes)
    """
    from .llm import _call_llm, _get_llm_config, call_llm_validated, LLMPublicationScoresResponse

    publications = _ensure_publications(_get_param(params, "publications"))
    keywords = _get_param(params, "keywords") or ""
    top_n = _get_param(params, "top_n") or 10
    llm_config = _get_param(params, "llm_config")

    config = _get_llm_config(ctx, llm_config)

    # Batch size for LLM calls - larger for local models to reduce overhead
    # Qwen 2.5 32B has 32K+ context, can handle 25-30 pubs easily (~2K tokens)
    # Cloud APIs are faster per-request so smaller batches with more parallelism works better
    is_local = config.get("provider") == "ollama"
    BATCH_SIZE = 25 if is_local else 10
    RELEVANCE_THRESHOLD = 40

    # Calculate total likes for each publication upfront
    for pub in publications:
        pub._total_likes = sum(p.likes_count for p in pub.posts) if pub.posts else 0

    # Create batches
    batches = []
    for i in range(0, len(publications), BATCH_SIZE):
        batches.append(publications[i:i + BATCH_SIZE])

    ctx.report_input({
        "publications_count": len(publications),
        "keywords": keywords,
        "top_n": top_n,
        "batch_size": BATCH_SIZE,
        "num_batches": len(batches),
        "strategy": f"Parallel LLM scoring ({len(batches)} batches), then filter threshold={RELEVANCE_THRESHOLD}, then N/2 relevant + N/2 popular",
        "llm_provider": config.get("provider"),
        "llm_model": config.get("model"),
    })

    async def score_batch(batch: List[DiscoveredPublication], batch_idx: int) -> List[dict]:
        """Score a batch of publications using LLM with Pydantic validation and retry."""
        # Build the prompt with all publications in this batch
        pubs_text = ""
        for i, pub in enumerate(batch):
            posts_titles = [p.title for p in pub.posts[:6]]
            posts_str = "; ".join(posts_titles) if posts_titles else "No posts"
            pubs_text += f"""
{i + 1}. {pub.name} (@{pub.handle})
   Description: {(pub.description or 'N/A')[:150]}
   Posts: {posts_str}
"""

        prompt = f"""Score these {len(batch)} Substack publications for relevance to: {keywords}

Publications:
{pubs_text}

For EACH publication, rate 0-100 where:
- 0-30: Not relevant (different topic entirely)
- 40-60: Somewhat relevant (occasionally covers the topic)
- 70-100: Highly relevant (core focus matches the keywords)

Respond with a JSON object containing exactly {len(batch)} scores, one per publication in order:
{{"scores": [
  {{"handle": "handle1", "score": 75, "reason": "brief explanation"}},
  {{"handle": "handle2", "score": 20, "reason": "brief explanation"}}
]}}"""

        try:
            # Use Pydantic-validated LLM call with retry on validation failure
            validated_response = await call_llm_validated(
                prompt=prompt,
                config=config,
                response_model=LLMPublicationScoresResponse,
                max_tokens=1000,
                max_retries=2,
            )
            scores = validated_response.scores

            # Map scores back to publications
            results = []
            for i, pub in enumerate(batch):
                if i < len(scores):
                    score_item = scores[i]
                    results.append({
                        "handle": pub.handle,
                        "score": float(score_item.score),
                        "reason": score_item.reason,
                    })
                else:
                    # LLM returned fewer scores than publications - fail loudly
                    logger.warning("llm_returned_fewer_scores", batch_idx=batch_idx, expected=len(batch), got=len(scores))
                    results.append({"handle": pub.handle, "score": 0, "reason": "missing from LLM response"})

            logger.info("batch_scored", batch_idx=batch_idx, count=len(results))
            return results

        except Exception as e:
            logger.error("batch_scoring_failed", batch_idx=batch_idx, error=str(e))
            # Fail loudly - LLM failures should not be masked with fake scores
            raise RuntimeError(f"LLM batch scoring failed for batch {batch_idx}: {e}") from e

    # Run batches with limited concurrency (Ollama can't handle too many parallel requests)
    # Use higher concurrency for cloud APIs, lower for local Ollama
    MAX_CONCURRENT = 3 if config.get("provider") == "ollama" else 10
    ctx.report_progress(5, f"Starting {len(batches)} LLM batches ({MAX_CONCURRENT} concurrent, 10 pubs each)...")

    # Track completion for progress reporting
    completed_batches = 0
    batch_results = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def score_batch_with_progress(batch, batch_idx):
        nonlocal completed_batches
        async with semaphore:
            result = await score_batch(batch, batch_idx)
        completed_batches += 1
        pct = int(10 + (completed_batches / len(batches) * 70))  # 10-80%
        ctx.report_progress(pct, f"Batch {completed_batches}/{len(batches)} done ({completed_batches * BATCH_SIZE} pubs scored)")
        return result

    # Run all batches concurrently (semaphore limits actual parallelism)
    tasks = [score_batch_with_progress(batch, i) for i, batch in enumerate(batches)]
    batch_results = await asyncio.gather(*tasks)

    ctx.report_progress(85, f"All {len(batches)} batches done, checking results...")

    # Flatten results and map back to publications
    handle_to_score = {}
    error_count = 0
    for batch_result in batch_results:
        for item in batch_result:
            handle_to_score[item["handle"]] = item
            if item.get("reason", "").startswith("error:"):
                error_count += 1

    # FAIL LOUDLY: If all publications had errors, the workflow should fail
    if error_count == len(publications):
        error_samples = [
            f"{item['handle']}: {item['reason']}"
            for item in list(handle_to_score.values())[:5]
        ]
        error_msg = f"All {len(publications)} LLM scoring requests failed. Samples: {error_samples}"
        logger.error("llm_scoring_all_failed", error_count=error_count, total=len(publications))
        ctx.report_output({
            "status": "error",
            "error": error_msg,
            "error_count": error_count,
            "total_publications": len(publications),
            "error_samples": error_samples,
        })
        raise RuntimeError(error_msg)

    # Warn if significant portion failed but continue
    if error_count > 0:
        logger.warning("llm_scoring_partial_failure", error_count=error_count, total=len(publications))
        ctx.report_progress(87, f"Warning: {error_count}/{len(publications)} scoring requests failed")

    # Apply scores to publications
    for pub in publications:
        score_info = handle_to_score.get(pub.handle, {})
        pub.llm_score = score_info.get("score", 0)
        pub._score_reason = score_info.get("reason", "")

    # Filter out irrelevant publications
    relevant_publications = [p for p in publications if (p.llm_score or 0) >= RELEVANCE_THRESHOLD]
    ruled_out = [p for p in publications if (p.llm_score or 0) < RELEVANCE_THRESHOLD]

    logger.info(
        "relevance_filter_applied",
        threshold=RELEVANCE_THRESHOLD,
        total=len(publications),
        relevant=len(relevant_publications),
        ruled_out=len(ruled_out),
    )

    # Selection: N/2 by relevance + N/2 by popularity (likes)
    # First get 2N candidates from relevant pool
    relevant_publications.sort(key=lambda p: p.llm_score or 0, reverse=True)
    candidate_pool = relevant_publications[:top_n * 2]

    n_by_relevance = top_n // 2
    n_by_likes = top_n - n_by_relevance

    # Top N/2 by relevance
    top_by_relevance = candidate_pool[:n_by_relevance]
    selected_handles = {p.handle for p in top_by_relevance}

    # Top N/2 by likes from remaining
    remaining = [p for p in candidate_pool if p.handle not in selected_handles]
    remaining.sort(key=lambda p: p._total_likes, reverse=True)
    top_by_likes = remaining[:n_by_likes]

    # Set final scores
    for p in top_by_relevance:
        p.final_score = p.llm_score
        p._selection_reason = "relevance"
    for p in top_by_likes:
        p.final_score = p.llm_score
        p._selection_reason = "popularity"

    # Combine and sort
    scored = top_by_relevance + top_by_likes
    scored.sort(key=lambda p: p.final_score or 0, reverse=True)

    # Stats
    all_llm_scores = [p.llm_score or 0 for p in publications]

    ctx.report_output({
        "status": "success",
        "total_publications": len(publications),
        "batches_processed": len(batches),
        "relevance_filter": {
            "threshold": RELEVANCE_THRESHOLD,
            "passed": len(relevant_publications),
            "ruled_out": len(ruled_out),
        },
        "candidate_pool_size": len(candidate_pool),
        "returned_count": len(scored),
        "selection_breakdown": {
            "by_relevance": len(top_by_relevance),
            "by_popularity": len(top_by_likes),
        },
        "score_distribution": {
            "min": round(min(all_llm_scores), 1) if all_llm_scores else 0,
            "max": round(max(all_llm_scores), 1) if all_llm_scores else 0,
            "avg": round(sum(all_llm_scores) / len(all_llm_scores), 1) if all_llm_scores else 0,
        },
        "selected_by_relevance": [
            {"handle": p.handle, "name": p.name, "score": round(p.llm_score or 0, 1), "likes": p._total_likes, "reason": getattr(p, '_score_reason', '')}
            for p in top_by_relevance
        ],
        "selected_by_popularity": [
            {"handle": p.handle, "name": p.name, "score": round(p.llm_score or 0, 1), "likes": p._total_likes, "reason": getattr(p, '_score_reason', '')}
            for p in top_by_likes
        ],
        "ruled_out": [
            {"handle": p.handle, "name": p.name, "score": round(p.llm_score or 0, 1), "reason": getattr(p, '_score_reason', '')}
            for p in sorted(ruled_out, key=lambda x: x.llm_score or 0, reverse=True)
        ],
        "all_scores": [
            {"handle": p.handle, "name": p.name, "score": round(p.llm_score or 0, 1), "likes": p._total_likes}
            for p in sorted(publications, key=lambda x: x.llm_score or 0, reverse=True)
        ],
        "llm_provider": config.get("provider"),
        "llm_model": config.get("model"),
    })

    return ScoreLLMRelevanceOutput(
        scored_publications=scored,
        status="success",
    )


async def generate_discovery_report(
    ctx,
    params: GenerateDiscoveryReportInput,
) -> GenerateDiscoveryReportOutput:
    """
    Generate a markdown report of discovered publications.
    """
    publications = _ensure_publications(
        _get_param(params, "publications") or
        _get_param(params, "scored_publications") or
        _get_param(params, "filtered_publications") or
        []
    )
    top_n = _get_param(params, "top_n") or 10
    category = _get_param(params, "category") or ""
    keywords = _get_param(params, "keywords") or ""

    ctx.report_input({
        "publications_count": len(publications),
        "top_n": top_n,
        "category": category,
        "keywords": keywords,
        "publications": [
            {"handle": p.handle, "name": p.name, "score": round(p.final_score or p.keyword_score or 0, 1)}
            for p in publications
        ],
    })

    # Take top N
    top_pubs = publications[:top_n]

    # Generate copy-paste ready handles list
    handles_list = ", ".join([p.handle for p in top_pubs])

    # Generate markdown report
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = f"""# Publication Discovery Report

**Category:** {category}
**Keywords:** {keywords}
**Generated:** {now}
**Results:** {len(top_pubs)} publications

## Source Handles

```
{handles_list}
```

---

"""

    for i, pub in enumerate(top_pubs, 1):
        score_str = ""
        if pub.final_score:
            score_str = f" (Score: {pub.final_score:.0f})"
        elif pub.keyword_score:
            score_str = f" (Keyword Score: {pub.keyword_score:.1f})"

        sub_str = ""
        if pub.subscriber_count:
            if pub.subscriber_count >= 1000000:
                sub_str = f"{pub.subscriber_count / 1000000:.1f}M"
            elif pub.subscriber_count >= 1000:
                sub_str = f"{pub.subscriber_count / 1000:.1f}K"
            else:
                sub_str = f"{pub.subscriber_count}"

        # Calculate total likes from posts
        total_likes = sum(p.likes_count for p in pub.posts) if pub.posts else 0
        likes_str = ""
        if total_likes >= 1000:
            likes_str = f"{total_likes / 1000:.1f}K"
        else:
            likes_str = str(total_likes)

        url = f"https://{pub.handle}.substack.com"
        if pub.custom_domain:
            url = f"https://{pub.custom_domain}"

        md += f"""## {i}. [{pub.name}]({url}){score_str}

**Handle:** @{pub.handle} | **Subscribers:** {sub_str or 'N/A'} | **Total Likes:** {likes_str} | **Type:** {pub.leaderboard_type}

{pub.description[:300] if pub.description else 'No description'}

"""

        if pub.posts:
            md += "**Sample Posts:**\n"
            for post in pub.posts[:3]:
                md += f"- [{post.title}]({post.url})"
                if post.likes_count:
                    md += f" ({post.likes_count} likes)"
                md += "\n"
            md += "\n"

        md += "---\n\n"

    # Build report data
    report_title = f"Discovery: {category} - {keywords[:30]}"
    report_tags = ["discovery", category] + [k.strip() for k in keywords.split(";")[:3]]

    report_json = {
        "category": category,
        "keywords": keywords,
        "total_results": len(top_pubs),
        "publications": [
            {
                "handle": p.handle,
                "name": p.name,
                "url": f"https://{p.handle}.substack.com",
                "subscriber_count": p.subscriber_count,
                "score": p.final_score or p.keyword_score,
            }
            for p in top_pubs
        ],
    }

    ctx.report_output({
        "status": "success",
        "report_title": report_title,
        "publications_count": len(top_pubs),
    })

    return GenerateDiscoveryReportOutput(
        report_title=report_title,
        report_markdown=md,
        report_tags=report_tags,
        report_json=report_json,
        status="success",
    )


# NOTE: The v3 vector-based discovery nodes (fetch_discovered_content, store_discovered_content,
# rank_publications) have been removed. The v4 LLM-based workflow (score_posts_llm,
# rank_publications_llm) is now the standard. Posts are stored via store_discovered_posts
# to the unified posts table.


async def generate_report(
    ctx,
    params: GenerateReportInput,
) -> GenerateReportOutput:
    """
    Generate a markdown report from Borda-ranked publications.

    Takes ALL ranked publications and generates a report with top N.
    """
    ranked_pubs = _get_param(params, "ranked_publications") or []
    top_n = _get_param(params, "top_n") or 10
    category = _get_param(params, "category") or ""
    keywords = _get_param(params, "keywords") or ""
    kv_key = _get_param(params, "kv_key") or "discovery_sources"

    # Convert dicts to ScoredPublicationResult if needed
    publications = []
    for pub in ranked_pubs:
        if isinstance(pub, dict):
            publications.append(ScoredPublicationResult(**pub))
        else:
            publications.append(pub)

    ctx.report_input({
        "publications_count": len(publications),
        "top_n": top_n,
        "category": category,
        "keywords": keywords,
        "publications": [
            {
                "handle": p.handle,
                "name": p.name,
                "borda_score": p.borda_score,
                "leaderboard": f"{p.leaderboard_type}#{p.leaderboard_rank}",
                "subscribers": p.subscriber_count,
                "total_likes": p.total_likes,
                "avg_relevance": round(p.avg_similarity, 3),
            }
            for p in publications
        ],
    })

    # Take top N
    top_pubs = publications[:top_n]

    # Generate copy-paste ready handles list
    handles_list = ", ".join([p.handle for p in top_pubs])

    # Helper to format numbers (0 = hidden/not available)
    def fmt_num(n, show_dash_for_zero=False):
        if n == 0 and show_dash_for_zero:
            return "-"
        if n >= 1000000:
            return f"{n / 1000000:.1f}M"
        elif n >= 1000:
            return f"{n / 1000:.1f}K"
        return str(n)

    # Calculate summary stats
    total_subscribers = sum(p.subscriber_count for p in top_pubs)
    total_likes = sum(p.total_likes for p in top_pubs)
    paid_count = len([p for p in top_pubs if p.leaderboard_type == "paid"])
    trending_count = len([p for p in top_pubs if p.leaderboard_type == "trending"])
    avg_relevance = sum(p.avg_similarity for p in top_pubs) / len(top_pubs) if top_pubs else 0

    # Generate markdown report
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = f"""# Publication Discovery Report

**Category:** {category}
**Keywords:** {keywords}
**Generated:** {now}

## Summary

| Metric | Value |
|--------|-------|
| Publications Found | {len(top_pubs)} |
| From Paid Leaderboard | {paid_count} |
| From Trending Leaderboard | {trending_count} |
| Total Combined Likes | {fmt_num(total_likes)} |
| Avg Relevance | {avg_relevance:.1%} |

### Ranking Method

**Borda Score** = rank_paid + rank_trending + 2×rank_avg_llm *(lower = better)*

- **rank_paid** — Position on Substack's paid leaderboard
- **rank_trending** — Position on Substack's trending leaderboard
- **rank_avg_llm** — Rank by average LLM relevance score across posts (counted 2x)

*LLM scores each post 0-100 for semantic relevance to your keywords.*

## Quick Copy - Handles

```
{handles_list}
```

## Rankings Table

| # | Publication | Borda | Leaderboard | Likes | Avg Relevance |
|---|-------------|-------|-------------|-------|---------------|
"""

    for i, pub in enumerate(top_pubs, 1):
        lb = f"{pub.leaderboard_type}#{pub.leaderboard_rank}"
        md += f"| {i} | [{pub.name}](https://{pub.handle}.substack.com) | {pub.borda_score} | {lb} | {fmt_num(pub.total_likes)} | {pub.avg_similarity:.0%} |\n"

    md += """
---

## Detailed Breakdown

"""

    for i, pub in enumerate(top_pubs, 1):
        url = f"https://{pub.handle}.substack.com"

        # Build rank breakdown string (4 signals)
        # Show actual penalty rank used (not null) so user sees what went into borda
        n_candidates = len(publications)
        penalty = n_candidates + 1
        rank_paid = pub.rank_by_paid if pub.rank_by_paid is not None else penalty
        rank_trending = pub.rank_by_trending if pub.rank_by_trending is not None else penalty
        rank_breakdown = f"paid={rank_paid} + trending={rank_trending} + avg_rel={pub.rank_by_avg_relevance} + max_rel={pub.rank_by_max_relevance}"

        subs_display = fmt_num(pub.subscriber_count, show_dash_for_zero=True)
        if pub.subscriber_count == 0:
            subs_display = "- (hidden)"

        md += f"""### {i}. [{pub.name}]({url})

- **Handle:** @{pub.handle}
- **Borda Score:** {pub.borda_score} ({rank_breakdown})
- **Leaderboard:** {pub.leaderboard_type} #{pub.leaderboard_rank}
- **Subscribers:** {subs_display}
- **Total Likes:** {fmt_num(pub.total_likes)}
- **Avg Relevance:** {pub.avg_similarity:.1%}

"""

    # Build report data
    report_title = f"Discovery: {category} - {keywords[:30]}"
    report_tags = ["discovery", "borda-ranking", category] + [k.strip() for k in keywords.split(";")[:3]]

    report_json = {
        "category": category,
        "keywords": keywords,
        "generated_at": now,
        "scoring_method": "borda_ranking",
        "borda_signals": ["rank_paid", "rank_trending", "rank_avg_relevance (2x)"],
        "summary": {
            "total_publications": len(top_pubs),
            "from_paid_leaderboard": paid_count,
            "from_trending_leaderboard": trending_count,
            "total_subscribers": total_subscribers,
            "total_likes": total_likes,
            "avg_relevance": round(avg_relevance, 3),
        },
        "handles_list": handles_list,
        "publications": [
            {
                "rank": i + 1,
                "handle": p.handle,
                "name": p.name,
                "url": f"https://{p.handle}.substack.com",
                "borda_score": p.borda_score,
                "leaderboard_type": p.leaderboard_type,
                "leaderboard_rank": p.leaderboard_rank,
                "subscriber_count": p.subscriber_count,
                "total_likes": p.total_likes,
                "avg_similarity": round(p.avg_similarity, 3),
                "ranks": {
                    "paid": p.rank_by_paid if p.rank_by_paid is not None else len(publications) + 1,
                    "trending": p.rank_by_trending if p.rank_by_trending is not None else len(publications) + 1,
                    "avg_relevance": p.rank_by_avg_relevance,
                    "max_relevance": p.rank_by_max_relevance,
                },
            }
            for i, p in enumerate(top_pubs)
        ],
    }

    # Store results in KV for other workflows to consume
    # Use the user-provided kv_key (static, predictable name)
    kv_value = {
        "category": category,
        "keywords": keywords,
        "generated_at": now,
        "handles": [p.handle for p in top_pubs],
        "publications": [
            {
                "handle": p.handle,
                "name": p.name,
                "borda_score": p.borda_score,
                "leaderboard_type": p.leaderboard_type,
                "leaderboard_rank": p.leaderboard_rank,
                "subscriber_count": p.subscriber_count,
                "total_likes": p.total_likes,
                "avg_relevance": round(p.avg_similarity, 3),
            }
            for p in top_pubs
        ],
    }
    # Store to KV - fail loudly if this doesn't work
    logger.info("generate_report_attempting_kv_put", key=kv_key, has_kv=hasattr(ctx, 'kv'), kv_client=getattr(ctx, '_kv_client', None))
    await ctx.kv.put(kv_key, kv_value)
    logger.info("discovery_results_stored_to_kv", key=kv_key, publications_count=len(top_pubs))

    ctx.report_output({
        "status": "success",
        "report_title": report_title,
        "publications_count": len(top_pubs),
        "handles_list": handles_list,
        "kv_key": kv_key,
        "publications": [
            {
                "rank": i + 1,
                "handle": p.handle,
                "name": p.name,
                "borda_score": p.borda_score,
                "leaderboard": f"{p.leaderboard_type}#{p.leaderboard_rank}",
                "subscribers": p.subscriber_count,
                "total_likes": p.total_likes,
                "avg_relevance": round(p.avg_similarity, 3),
                "max_relevance": round(p.max_similarity, 3),
            }
            for i, p in enumerate(top_pubs)
        ],
    })

    return GenerateReportOutput(
        report_title=report_title,
        report_markdown=md,
        report_tags=report_tags,
        report_json=report_json,
        status="success",
    )


# =============================================================================
# LLM-BASED RELEVANCE SCORING (New Approach)
# =============================================================================

def _normalize_keywords(keywords: str) -> str:
    """Normalize keywords for cache key: lowercase, sorted, semicolon-joined."""
    parts = [k.strip().lower() for k in keywords.split(";") if k.strip()]
    parts.sort()
    return ";".join(parts)


def _hash_prompt(prompt: str) -> str:
    """Create a short hash of the scoring prompt for cache key."""
    import hashlib
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


async def _get_cached_post_scores(post_urls: List[str], keywords_normalized: str, model: str, ttl_hours: int = 48) -> Dict[str, dict]:
    """
    Get cached LLM relevance scores for posts.

    Cache key includes model so different models don't share scores.
    TTL default is 48 hours since scores are model-specific.

    Returns: {post_url: {score, publication_handle}} for cached entries
    """
    from shared.database import get_db_session
    from sqlalchemy import text
    from datetime import datetime, timezone, timedelta

    cached = {}
    cutoff = datetime.now(timezone.utc) - timedelta(hours=ttl_hours)

    async with get_db_session() as db:
        result = await db.execute(
            text("""
                SELECT post_url, score, publication_handle
                FROM llm_relevance_cache
                WHERE post_url = ANY(:urls)
                  AND keywords_normalized = :keywords
                  AND model = :model
                  AND cached_at > :cutoff
            """),
            {"urls": post_urls, "keywords": keywords_normalized, "model": model, "cutoff": cutoff}
        )
        for row in result.fetchall():
            cached[row[0]] = {
                "score": row[1],
                "publication_handle": row[2],
            }

    return cached


async def _store_post_scores(scores: List[dict], keywords_normalized: str, model: str):
    """
    Store LLM relevance scores for posts in cache.

    scores: [{url, score, pub_handle}, ...]
    model: LLM model used for scoring (part of cache key)
    """
    from shared.database import get_db_session
    from sqlalchemy import text

    if not scores:
        return

    async with get_db_session() as db:
        for s in scores:
            await db.execute(
                text("""
                    INSERT INTO llm_relevance_cache
                        (id, post_url, keywords_normalized, model, score, publication_handle, cached_at)
                    VALUES
                        (gen_random_uuid(), :url, :keywords, :model, :score, :pub_handle, NOW())
                    ON CONFLICT (post_url, keywords_normalized, model)
                    DO UPDATE SET
                        score = EXCLUDED.score,
                        publication_handle = EXCLUDED.publication_handle,
                        cached_at = NOW()
                """),
                {
                    "url": s["url"],
                    "keywords": keywords_normalized,
                    "model": model,
                    "score": s["score"],
                    "pub_handle": s["pub_handle"],
                }
            )
        await db.commit()


async def score_posts_llm(
    ctx,
    params: ScorePostsLLMInput,
) -> ScorePostsLLMOutput:
    """
    Score all posts using LLM for relevance to keywords.

    Instead of vector embeddings, we send title+subtitle of ALL posts to an LLM
    and ask it to score each one 0-100 for relevance. This is more accurate
    because the LLM understands that "How Netflix Handles 1M Requests" is
    about system design even if it doesn't contain those exact words.

    Strategy:
    - Check cache for publications already scored with same keywords
    - Collect posts only from uncached publications
    - Send to LLM in batches of ~100 posts per call
    - LLM returns scores for each post
    - Store results in cache
    - Aggregate scores per publication (avg, max)

    Cache TTL: 1 hour
    """
    from .llm import _call_llm, _get_llm_config, call_llm_validated, LLMScoresResponse
    from .schemas import ScorePostsLLMOutput, ScoredPublication, PostScore

    # Get publications from params, falling back to ctx.state if empty
    # (Template resolution {{publications}} sometimes fails to pass data correctly)
    raw_pubs = _get_param(params, "publications")
    if not raw_pubs and hasattr(ctx, 'state'):
        state_pubs = ctx.state.get("publications", [])
        if state_pubs:
            logger.info("score_posts_llm_using_ctx_state_publications", count=len(state_pubs))
            raw_pubs = state_pubs

    publications = _ensure_publications(raw_pubs)
    keywords = _get_param(params, "keywords") or ""
    scoring_prompt = _get_param(params, "scoring_prompt") or ""
    llm_config = _get_param(params, "llm_config")
    llm_retries = _get_param(params, "llm_retries")
    if llm_retries is None:
        llm_retries = 1  # Default: 1 retry

    config = _get_llm_config(ctx, llm_config)
    keywords_normalized = _normalize_keywords(keywords)
    base_model = config.get("model", "unknown")
    prompt_hash = _hash_prompt(scoring_prompt)
    # Cache key includes model + prompt hash so different prompts don't share scores
    cache_model_key = f"{base_model}:{prompt_hash}"

    # Collect ALL posts with their publication info
    all_posts = []
    for pub in publications:
        for post in pub.posts:
            all_posts.append({
                "pub_handle": pub.handle,
                "pub_name": pub.name,
                "url": post.url,
                "title": post.title,
                "subtitle": post.subtitle or "",  # Already merged subtitle/description
                "tags": post.tags if hasattr(post, 'tags') else [],
                "section_name": post.section_name if hasattr(post, 'section_name') else "",
            })

    # Check cache for already-scored POSTS (not publications)
    # Cache key includes model + prompt_hash so different prompts have separate caches
    all_urls = [p["url"] for p in all_posts]
    cached_post_scores = await _get_cached_post_scores(all_urls, keywords_normalized, cache_model_key)
    cached_urls = set(cached_post_scores.keys())

    # Filter to only uncached posts
    uncached_posts = [p for p in all_posts if p["url"] not in cached_urls]

    # Sample posts to show what data is available for LLM (only include non-empty fields)
    sample_posts_for_llm = []
    for p in all_posts[:3]:  # Show first 3 posts
        sample = {"title": p["title"][:100]}
        if p["subtitle"]:
            sample["subtitle"] = p["subtitle"][:200]
        if p.get("tags"):
            sample["tags"] = p["tags"]
        if p.get("section_name"):
            sample["section_name"] = p["section_name"]
        sample_posts_for_llm.append(sample)

    ctx.report_input({
        "scoring_prompt": scoring_prompt,
        "keywords": keywords,
        "keywords_normalized": keywords_normalized,
        "llm_provider": config.get("provider"),
        "llm_model": base_model,
        "llm_retries": llm_retries,
        "prompt_hash": prompt_hash,
        "cache_key": f"{keywords_normalized}|{cache_model_key}",
        "publications_count": len(publications),
        "total_posts": len(all_posts),
        "cached_posts": len(cached_urls),
        "uncached_posts": len(uncached_posts),
        "sample_posts_for_llm": sample_posts_for_llm,
        "prompt_fields_used": ["title", "subtitle", "tags", "section_name"],
    })

    if not all_posts:
        ctx.report_output({
            "status": "no_posts",
            "message": "No posts to score",
        })
        return ScorePostsLLMOutput(status="no_posts").model_dump()

    # If everything is cached, skip LLM calls entirely
    if not uncached_posts:
        ctx.report_progress(100, f"All {len(cached_urls)} posts found in cache!")
        batches = []
    else:
        # Batch UNCACHED posts for LLM calls
        # Gemini 2.0 Flash handles larger batches well - fewer API calls = faster + cheaper
        # Local LLMs (Ollama) need smaller batches to avoid timeouts
        is_local = config.get("provider") == "ollama"
        BATCH_SIZE = 15 if is_local else 40
        batches = []
        for i in range(0, len(uncached_posts), BATCH_SIZE):
            batches.append(uncached_posts[i:i + BATCH_SIZE])

        ctx.report_progress(5, f"Scoring {len(uncached_posts)} posts in {len(batches)} batches ({len(cached_urls)} cached)...")

    async def score_batch(batch: List[dict], batch_idx: int) -> List[dict]:
        """Score a batch of posts using LLM with Pydantic validation and retry."""
        # Build the posts list for the prompt
        # Include title, subtitle, tags, section_name for scoring
        posts_text = ""
        for i, post in enumerate(batch):
            title = post["title"][:200]  # Truncate long titles (max seen: 177)
            subtitle = post["subtitle"][:350] if post["subtitle"] else ""  # Merged subtitle/description
            tags = post.get("tags", [])
            section_name = post.get("section_name", "")

            posts_text += f"{i+1}. **{title}**"
            if subtitle:
                posts_text += f"\n   Subtitle: {subtitle}"
            if tags:
                posts_text += f"\n   Tags: {', '.join(tags)}"
            if section_name:
                posts_text += f"\n   Section: {section_name}"
            posts_text += "\n\n"

        prompt = f"""{scoring_prompt}

Keywords: "{keywords}"

Posts to score (total: {len(batch)}):
{posts_text}

Respond with a JSON object containing exactly {len(batch)} integer scores (0-100), one per post in order:
{{"scores": [85, 20, 70, 45, ...]}}"""

        # Log the first batch's full prompt for debugging
        if batch_idx == 0:
            logger.info("llm_prompt_batch_0", prompt_length=len(prompt), prompt=prompt)

        # Retry logic for network/timeout errors
        last_error = None
        for attempt in range(llm_retries + 1):
            try:
                # Use Pydantic-validated LLM call with retry on validation failure
                validated_response = await call_llm_validated(
                    prompt=prompt,
                    config=config,
                    response_model=LLMScoresResponse,
                    max_tokens=2000,
                    max_retries=2,
                )
                scores = validated_response.scores

                # Ensure we have enough scores (pad if LLM returned fewer)
                while len(scores) < len(batch):
                    logger.warning("llm_returned_fewer_scores", batch_idx=batch_idx, expected=len(batch), got=len(scores))
                    scores.append(0)

                # Map scores back to posts
                results = []
                for i, post in enumerate(batch):
                    score = int(scores[i]) if i < len(scores) else 0
                    score = max(0, min(100, score))  # Clamp to 0-100
                    results.append({
                        "url": post["url"],
                        "pub_handle": post["pub_handle"],
                        "score": score,
                    })
                    # Log each score for real-time monitoring
                    logger.info(
                        "post_scored",
                        pub=post["pub_handle"],
                        score=score,
                        title=post["title"][:60],
                    )

                logger.info("batch_scored", batch_idx=batch_idx, count=len(results), attempt=attempt + 1)
                return results

            except Exception as e:
                last_error = e
                if attempt < llm_retries:
                    wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s...
                    logger.warning("batch_scoring_retry", batch_idx=batch_idx, attempt=attempt + 1, max_retries=llm_retries, error=str(e), wait_time=wait_time)
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("batch_scoring_failed", batch_idx=batch_idx, error=str(e), attempts=attempt + 1)

        # All retries exhausted
        raise RuntimeError(f"LLM batch scoring failed for batch {batch_idx} after {llm_retries + 1} attempts: {last_error}") from last_error

    # Run LLM scoring for uncached publications
    fresh_scores = []
    if all_posts:
        # Run batches with limited concurrency
        # Keep conservative to avoid rate limits (especially Gemini free tier)
        # Local LLMs (Ollama) need lower concurrency to avoid timeouts
        MAX_CONCURRENT = 2 if is_local else 3
        completed_batches = set()  # Track which batches are done (thread-safe set operations)

        semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        async def score_with_progress(batch, batch_idx):
            async with semaphore:
                result = await score_batch(batch, batch_idx)
            # Store batch results immediately so we don't lose them if later batches fail
            await _store_post_scores(result, keywords_normalized, cache_model_key)
            fresh_scores.extend(result)
            completed_batches.add(batch_idx)
            completed_count = len(completed_batches)
            pct = int(10 + (completed_count / len(batches) * 80))
            ctx.report_progress(pct, f"Batch {completed_count}/{len(batches)} done ({len(fresh_scores)} scored)")
            return result

        tasks = [score_with_progress(batch, i) for i, batch in enumerate(batches)]
        await asyncio.gather(*tasks)

    ctx.report_progress(90, "Aggregating scores per publication...")

    # Combine cached + fresh scores into a single lookup by URL
    all_post_scores: Dict[str, dict] = {}

    # Add cached scores
    for url, data in cached_post_scores.items():
        all_post_scores[url] = {"score": data["score"], "pub_handle": data["publication_handle"]}

    # Add fresh scores (may overlap with cached for same URL, fresh wins)
    for score_item in fresh_scores:
        all_post_scores[score_item["url"]] = {"score": score_item["score"], "pub_handle": score_item["pub_handle"]}

    # Build scored publications by aggregating post scores per publication
    scored_publications = []
    total_posts_scored = 0

    for pub in publications:
        total_likes = sum(p.likes_count for p in pub.posts)

        # Collect scores for all posts in this publication
        pub_post_scores = []
        for post in pub.posts:
            if post.url in all_post_scores:
                pub_post_scores.append({
                    "url": post.url,
                    "score": all_post_scores[post.url]["score"],
                })

        if not pub_post_scores:
            continue

        score_values = [s["score"] for s in pub_post_scores]
        scored_publications.append(ScoredPublication(
            handle=pub.handle,
            name=pub.name,
            subscriber_count=pub.subscriber_count,
            leaderboard_type=pub.leaderboard_type,
            leaderboard_rank=pub.leaderboard_rank,
            post_scores=[PostScore(url=s["url"], score=s["score"]) for s in pub_post_scores],
            avg_score=round(sum(score_values) / len(score_values), 1) if score_values else 0,
            max_score=max(score_values) if score_values else 0,
            total_likes=total_likes,
        ))
        total_posts_scored += len(pub_post_scores)

    # Sort by avg_score descending for the output
    scored_publications.sort(key=lambda p: p.avg_score, reverse=True)

    ctx.report_output({
        "status": "success",
        "total_posts_scored": total_posts_scored,
        "publications_scored": len(scored_publications),
        "posts_from_cache": len(cached_urls),
        "posts_freshly_scored": len(fresh_scores),
        "batches_processed": len(batches),
        "top_by_avg_score": [
            {"handle": p.handle, "name": p.name, "avg_score": p.avg_score, "max_score": p.max_score}
            for p in scored_publications[:20]
        ],
        "score_distribution": {
            "avg_of_avgs": round(sum(p.avg_score for p in scored_publications) / len(scored_publications), 1) if scored_publications else 0,
            "max_score_seen": max(p.max_score for p in scored_publications) if scored_publications else 0,
        },
    })

    return ScorePostsLLMOutput(
        scored_publications=scored_publications,
        total_posts_scored=total_posts_scored,
        status="success",
    ).model_dump()


async def rank_publications_llm(
    ctx,
    params: RankPublicationsLLMInput,
) -> RankPublicationsLLMOutput:
    """
    Rank publications using Borda scoring with LLM relevance scores.

    Uses 4 signals (avg counted twice to weight relevance more):
    - Paid leaderboard rank
    - Trending leaderboard rank
    - Avg LLM relevance score (counted twice)

    Returns ALL publications ranked by Borda score (lowest = best).
    """
    scored_pubs = _get_param(params, "scored_publications") or []

    # Convert to dicts for easier manipulation
    publications = []
    for pub in scored_pubs:
        if isinstance(pub, dict):
            publications.append(pub)
        else:
            publications.append(pub.model_dump())

    ctx.report_input({
        "publications_count": len(publications),
        "ranking_method": "borda (paid_rank + trending_rank + 2×avg_llm_score)",
    })

    if not publications:
        ctx.report_output({
            "status": "no_publications",
            "message": "No publications to rank",
        })
        return RankPublicationsOutput(status="no_publications").model_dump()

    n_candidates = len(publications)
    penalty_rank = n_candidates + 1

    # Rank by paid leaderboard (lower leaderboard_rank = better)
    paid_pubs = sorted(
        [p for p in publications if p.get("leaderboard_type") == "paid"],
        key=lambda p: p.get("leaderboard_rank", 999)
    )
    for i, p in enumerate(paid_pubs, 1):
        p["rank_by_paid"] = i

    # Rank by trending leaderboard
    trending_pubs = sorted(
        [p for p in publications if p.get("leaderboard_type") == "trending"],
        key=lambda p: p.get("leaderboard_rank", 999)
    )
    for i, p in enumerate(trending_pubs, 1):
        p["rank_by_trending"] = i

    # Rank by avg_score (higher = better, so reverse)
    by_avg = sorted(publications, key=lambda p: p.get("avg_score", 0), reverse=True)
    for i, p in enumerate(by_avg, 1):
        p["rank_by_avg_relevance"] = i

    # Calculate Borda score (avg counted twice to weight relevance more)
    results = []
    for p in publications:
        paid_rank = p.get("rank_by_paid") or penalty_rank
        trending_rank = p.get("rank_by_trending") or penalty_rank
        rank_avg = p.get("rank_by_avg_relevance", penalty_rank)

        # avg_llm counted twice: paid + trending + avg + avg
        borda_score = paid_rank + trending_rank + rank_avg + rank_avg

        post_scores = p.get("post_scores", [])

        results.append(ScoredPublicationResult(
            handle=p.get("handle", ""),
            name=p.get("name", ""),
            subscriber_count=p.get("subscriber_count", 0),
            matching_posts=len(post_scores),
            avg_similarity=p.get("avg_score", 0) / 100.0,  # Convert to 0-1 for compatibility
            max_similarity=p.get("max_score", 0) / 100.0,  # Still stored for reference, not used in ranking
            total_likes=p.get("total_likes", 0),
            leaderboard_type=p.get("leaderboard_type", ""),
            leaderboard_rank=p.get("leaderboard_rank", 0),
            borda_score=borda_score,
            rank_by_paid=p.get("rank_by_paid"),
            rank_by_trending=p.get("rank_by_trending"),
            rank_by_avg_relevance=rank_avg,
            rank_by_max_relevance=rank_avg,  # Same as avg (counted twice)
            selection_reason=f"borda={borda_score}",
        ))

    # Sort by Borda score (lowest = best)
    results.sort(key=lambda r: r.borda_score)

    ctx.report_output({
        "status": "success",
        "ranking_method": "borda_llm",
        "borda_signals": ["rank_paid", "rank_trending", "rank_avg_llm_score", "rank_avg_llm_score"],
        "borda_formula": "paid + trending + avg + avg (relevance weighted 2x)",
        "penalty_rank": penalty_rank,
        "total_ranked": len(results),
        "ranked_publications": [
            {
                "rank": i + 1,
                "handle": r.handle,
                "name": r.name,
                "borda_score": r.borda_score,
                "rank_paid": r.rank_by_paid if r.rank_by_paid else penalty_rank,
                "rank_trending": r.rank_by_trending if r.rank_by_trending else penalty_rank,
                "rank_avg_relevance": r.rank_by_avg_relevance,
                "leaderboard": f"{r.leaderboard_type}#{r.leaderboard_rank}",
                "avg_llm_score": round(r.avg_similarity * 100),
                "max_llm_score": round(r.max_similarity * 100),
                "total_likes": r.total_likes,
            }
            for i, r in enumerate(results)
        ],
    })

    return RankPublicationsOutput(
        ranked_publications=results,
        total_in_category=len(results),
        status="success",
    ).model_dump()
