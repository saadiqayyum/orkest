"""
Database operations for the AI Substack Mirroring Engine.

These are the workflow node functions for database interactions.
All domain-specific queries live here, keeping business logic in the team repo.
"""
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
from uuid import UUID
import structlog

from sqlalchemy import select, func, and_, or_, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import attributes

from shared.database import get_db_session
from shared.models import TargetItem, ProductionQueueItem, QueueStatus, Post, IdeaSourcePost
from .schemas import (
    # Target items
    InsertTargetItemsInput, InsertTargetItemsOutput,
    # Production queue
    QueueGapIdeasInput, QueueGapIdeasOutput, QueuedIdea,
    GetExistingTopicsInput, GetExistingTopicsOutput,
    GetPendingQueueItemsInput, GetPendingQueueItemsOutput, PendingQueueItem,
    LoadDraftingContextInput, LoadDraftingContextOutput,
    UpdateQueueItemDraftInput, UpdateQueueItemDraftOutput,
    SetQueueItemDraftingInput, SetQueueItemDraftingOutput,
    ResetDraftingStatusInput, ResetDraftingStatusOutput,
    ResetStaleDraftingItemsInput, ResetStaleDraftingItemsOutput,
    GetQueueItemForPublishInput, GetQueueItemForPublishOutput, QueueItemForPublish,
    FinalizePublicationInput, FinalizePublicationOutput,
    # Config
    LoadTargetConfigInput, LoadTargetConfigOutput,
    # Unified posts schemas
    UpsertPostsInput, UpsertPostsOutput, UpsertedPost, PostData,
    GetPostsInput, GetPostsOutput, PostResult,
    GetPostsNeedingContentInput, GetPostsNeedingContentOutput,
    UpdatePostMetricsInput, UpdatePostMetricsOutput,
    IdentifyNewPostUrlsInput, IdentifyNewPostUrlsOutput,
    GetPostsNeedingEmbeddingsInput, GetPostsNeedingEmbeddingsOutput, PostNeedingEmbedding,
    UpdatePostEmbeddingsInput, UpdatePostEmbeddingsOutput,
    QueueSourcesFromPostsInput, QueueSourcesFromPostsOutput, QueuedSourceItem,
    # Idea generation
    ResolveSourcePublicationsInput, ResolveSourcePublicationsOutput,
    FetchTopSourcePostsInput, FetchTopSourcePostsOutput, SourcePost,
    FetchMyPublishedPostsInput, FetchMyPublishedPostsOutput, MyPublishedPost,
    FilterWorkedSourcePostsInput, FilterWorkedSourcePostsOutput,
)

logger = structlog.get_logger()


# =============================================================================
# UNIFIED POSTS OPERATIONS
# =============================================================================

async def upsert_posts(
    ctx,
    params: UpsertPostsInput,
) -> UpsertPostsOutput:
    """
    Insert or update posts in the unified posts table.

    Deduplicates by URL - if URL exists, updates the record.
    Posts are global - they belong to a publication_handle, NOT a target.

    Input:
    - posts: List of PostData objects
    - discovery_source: Source of these posts ('content_sync', 'publication_discovery')

    Output:
    - inserted_count: Number of new posts inserted
    - updated_count: Number of existing posts updated
    - upserted_posts: List of all upserted posts with their IDs
    """
    posts = params.posts
    discovery_source = params.discovery_source

    ctx.report_input({
        "posts_count": len(posts),
        "discovery_source": discovery_source,
        "sample_posts": [
            {"url": p.url, "title": p.title[:50] if p.title else "", "publication_handle": p.publication_handle}
            for p in posts[:5]
        ],
    })

    if not posts:
        ctx.report_output({
            "inserted_count": 0,
            "updated_count": 0,
            "status": "success",
            "message": "No posts to upsert",
        })
        return UpsertPostsOutput(
            inserted_count=0,
            updated_count=0,
            upserted_posts=[],
            status="success",
        )

    async with get_db_session() as db:
        upserted_posts = []
        inserted_count = 0
        updated_count = 0

        for post_data in posts:
            # Check if post exists
            result = await db.execute(
                select(Post).where(Post.url == post_data.url)
            )
            existing_post = result.scalar_one_or_none()

            # Parse published_at if string
            published_at = post_data.published_at
            if isinstance(published_at, str):
                published_at = datetime.fromisoformat(published_at.replace("Z", "+00:00"))

            if existing_post:
                # Update existing post
                existing_post.title = post_data.title
                existing_post.publication_handle = post_data.publication_handle
                if post_data.subtitle:
                    existing_post.subtitle = post_data.subtitle

                # IMPORTANT: Never overwrite full content with partial content
                # Only update content if:
                # 1. New data has content AND
                # 2. Either existing has no content OR new data is full content OR existing is partial
                if post_data.content_raw:
                    should_update_content = (
                        not existing_post.content_raw or  # No existing content
                        post_data.has_full_content or     # New content is full
                        not existing_post.has_full_content  # Existing is partial, so OK to overwrite
                    )
                    if should_update_content:
                        existing_post.content_raw = post_data.content_raw
                        existing_post.content_fetched_at = datetime.now(timezone.utc)
                        existing_post.has_full_content = post_data.has_full_content
                    else:
                        logger.debug(
                            "skipped_content_overwrite",
                            url=post_data.url,
                            reason="existing has full content, new is partial"
                        )

                if post_data.author:
                    existing_post.author = post_data.author
                if post_data.substack_post_id:
                    existing_post.substack_post_id = post_data.substack_post_id
                if post_data.slug:
                    existing_post.slug = post_data.slug
                if post_data.post_type:
                    existing_post.post_type = post_data.post_type
                if post_data.audience:
                    existing_post.audience = post_data.audience
                if post_data.word_count:
                    existing_post.word_count = post_data.word_count
                if post_data.tags:
                    existing_post.tags = post_data.tags
                if post_data.section_name:
                    existing_post.section_name = post_data.section_name
                existing_post.is_paywalled = post_data.is_paywalled
                # Note: has_full_content is updated above with content_raw
                # Update metrics if provided
                if post_data.likes_count > 0 or existing_post.likes_count == 0:
                    existing_post.likes_count = post_data.likes_count
                if post_data.comments_count > 0 or existing_post.comments_count == 0:
                    existing_post.comments_count = post_data.comments_count
                if post_data.shares_count > 0 or existing_post.shares_count == 0:
                    existing_post.shares_count = post_data.shares_count
                if published_at:
                    existing_post.published_at = published_at
                existing_post.metrics_updated_at = datetime.now(timezone.utc)

                upserted_posts.append(UpsertedPost(
                    id=str(existing_post.id),
                    url=post_data.url,
                    title=post_data.title,
                    publication_handle=post_data.publication_handle,
                    is_new=False,
                ))
                updated_count += 1

            else:
                # Insert new post
                new_post = Post(
                    url=post_data.url,
                    publication_handle=post_data.publication_handle,
                    title=post_data.title,
                    subtitle=post_data.subtitle,
                    content_raw=post_data.content_raw,
                    author=post_data.author,
                    substack_post_id=post_data.substack_post_id,
                    slug=post_data.slug,
                    post_type=post_data.post_type,
                    audience=post_data.audience,
                    word_count=post_data.word_count,
                    tags=post_data.tags if post_data.tags else None,
                    section_name=post_data.section_name,
                    is_paywalled=post_data.is_paywalled,
                    has_full_content=post_data.has_full_content,
                    likes_count=post_data.likes_count,
                    comments_count=post_data.comments_count,
                    shares_count=post_data.shares_count,
                    published_at=published_at,
                    discovery_source=discovery_source,
                    content_fetched_at=datetime.now(timezone.utc) if post_data.content_raw else None,
                )
                db.add(new_post)
                await db.flush()

                upserted_posts.append(UpsertedPost(
                    id=str(new_post.id),
                    url=post_data.url,
                    title=post_data.title,
                    publication_handle=post_data.publication_handle,
                    is_new=True,
                ))
                inserted_count += 1

        await db.commit()

        logger.info(
            "posts_upserted",
            inserted=inserted_count,
            updated=updated_count,
            discovery_source=discovery_source,
        )

        ctx.report_output({
            "inserted_count": inserted_count,
            "updated_count": updated_count,
            "total_upserted": len(upserted_posts),
            "sample_upserted": [p.model_dump() for p in upserted_posts[:5]],
            "status": "success",
        })

        return UpsertPostsOutput(
            inserted_count=inserted_count,
            updated_count=updated_count,
            upserted_posts=upserted_posts,
            status="success",
        )


async def identify_new_post_urls(
    ctx,
    params: IdentifyNewPostUrlsInput,
) -> IdentifyNewPostUrlsOutput:
    """
    Identify which URLs are not yet in the posts table.

    Input:
    - urls: List of URLs to check
    - posts: Full post objects (optional, for returning full objects for new URLs)

    Output:
    - new_urls: URLs not in the database
    - existing_urls: URLs already in the database
    - new_posts: Full post objects for new URLs (if posts param was provided)
    """
    urls = params.urls or []
    posts = params.posts or []

    # Build URL-to-post mapping
    posts_by_url = {p.get("url") if isinstance(p, dict) else p.url: p for p in posts}

    # If we have posts but not urls, extract urls from posts
    if posts and not urls:
        urls = list(posts_by_url.keys())

    ctx.report_input({
        "urls_count": len(urls),
        "posts_count": len(posts),
    })

    if not urls:
        ctx.report_output({
            "new_urls": [],
            "existing_urls": [],
            "status": "success",
        })
        return IdentifyNewPostUrlsOutput(
            new_urls=[],
            existing_urls=[],
            new_posts=[],
            status="success",
        )

    async with get_db_session() as db:
        # Find existing URLs
        result = await db.execute(
            select(Post.url).where(Post.url.in_(urls))
        )
        existing_urls = set(row[0] for row in result.fetchall())

        new_urls = [url for url in urls if url not in existing_urls]
        new_posts = [posts_by_url[url] for url in new_urls if url in posts_by_url]

        logger.info(
            "new_post_urls_identified",
            total=len(urls),
            new=len(new_urls),
            existing=len(existing_urls),
        )

        ctx.report_output({
            "new_urls_count": len(new_urls),
            "existing_urls_count": len(existing_urls),
            "sample_new_urls": new_urls[:5],
            "status": "success",
        })

        return IdentifyNewPostUrlsOutput(
            new_urls=new_urls,
            existing_urls=list(existing_urls),
            new_posts=new_posts,
            status="success",
        )


async def get_posts(
    ctx,
    params: GetPostsInput,
) -> GetPostsOutput:
    """
    Query posts from the unified posts table.

    Input:
    - publication_handles: Filter by publication handles
    - min_likes: Minimum likes threshold
    - max_age_days: Max age in days
    - has_content: Filter by content availability
    - has_embedding: Filter by embedding availability
    - limit: Max posts to return
    - offset: Offset for pagination
    - order_by: 'likes_count' or 'published_at'

    Output:
    - posts: List of PostResult
    - count: Number of posts returned
    - total_count: Total matching posts (before limit)
    """
    publication_handles = params.publication_handles
    min_likes = params.min_likes
    max_age_days = params.max_age_days
    has_content = params.has_content
    has_embedding = params.has_embedding
    limit = params.limit
    offset = params.offset
    order_by = params.order_by

    ctx.report_input({
        "publication_handles": publication_handles,
        "min_likes": min_likes,
        "max_age_days": max_age_days,
        "has_content": has_content,
        "has_embedding": has_embedding,
        "limit": limit,
        "offset": offset,
        "order_by": order_by,
    })

    async with get_db_session() as db:
        # Build query
        conditions = []

        if publication_handles:
            conditions.append(Post.publication_handle.in_(publication_handles))

        if min_likes > 0:
            conditions.append(Post.likes_count >= min_likes)

        if max_age_days:
            cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
            conditions.append(Post.published_at >= cutoff)

        if has_content is True:
            conditions.append(Post.content_raw.isnot(None))
        elif has_content is False:
            conditions.append(Post.content_raw.is_(None))

        if has_embedding is True:
            conditions.append(Post.embedding.isnot(None))
        elif has_embedding is False:
            conditions.append(Post.embedding.is_(None))

        # Get total count
        count_query = select(func.count(Post.id))
        if conditions:
            count_query = count_query.where(and_(*conditions))
        total_result = await db.execute(count_query)
        total_count = total_result.scalar()

        # Build main query
        query = select(Post)
        if conditions:
            query = query.where(and_(*conditions))

        # Order by
        if order_by == "published_at":
            query = query.order_by(Post.published_at.desc())
        else:
            query = query.order_by(Post.likes_count.desc())

        # Pagination
        query = query.offset(offset).limit(limit)

        result = await db.execute(query)
        posts = result.scalars().all()

        post_results = [
            PostResult(
                id=str(p.id),
                url=p.url,
                publication_handle=p.publication_handle,
                title=p.title,
                subtitle=p.subtitle,
                author=p.author,
                likes_count=p.likes_count or 0,
                comments_count=p.comments_count or 0,
                shares_count=p.shares_count or 0,
                published_at=p.published_at.isoformat() if p.published_at else None,
                has_content=bool(p.content_raw),
                has_embedding=bool(p.embedding is not None),
                is_paywalled=p.is_paywalled or False,
                has_full_content=p.has_full_content if p.has_full_content is not None else True,
            )
            for p in posts
        ]

        logger.info(
            "posts_queried",
            count=len(post_results),
            total=total_count,
        )

        ctx.report_output({
            "count": len(post_results),
            "total_count": total_count,
            "sample_posts": [p.model_dump() for p in post_results[:5]],
            "status": "success",
        })

        return GetPostsOutput(
            posts=post_results,
            count=len(post_results),
            total_count=total_count,
            status="success",
        )


async def get_posts_needing_content(
    ctx,
    params: GetPostsNeedingContentInput,
) -> GetPostsNeedingContentOutput:
    """
    Get posts that don't have full content yet.

    Returns posts where:
    - content_raw is NULL, or
    - has_full_content is FALSE (paywalled content we couldn't access)
    """
    publication_handles = params.publication_handles
    limit = params.limit

    ctx.report_input({
        "publication_handles": publication_handles,
        "limit": limit,
    })

    async with get_db_session() as db:
        conditions = [
            or_(
                Post.content_raw.is_(None),
                Post.has_full_content == False,
            )
        ]

        if publication_handles:
            conditions.append(Post.publication_handle.in_(publication_handles))

        result = await db.execute(
            select(Post)
            .where(and_(*conditions))
            .order_by(Post.likes_count.desc())
            .limit(limit)
        )
        posts = result.scalars().all()

        post_results = [
            PostResult(
                id=str(p.id),
                url=p.url,
                publication_handle=p.publication_handle,
                title=p.title,
                subtitle=p.subtitle,
                author=p.author,
                likes_count=p.likes_count or 0,
                comments_count=p.comments_count or 0,
                shares_count=p.shares_count or 0,
                published_at=p.published_at.isoformat() if p.published_at else None,
                has_content=bool(p.content_raw),
                has_embedding=bool(p.embedding is not None),
                is_paywalled=p.is_paywalled or False,
                has_full_content=p.has_full_content if p.has_full_content is not None else True,
            )
            for p in posts
        ]

        logger.info("posts_needing_content_found", count=len(post_results))

        ctx.report_output({
            "count": len(post_results),
            "sample_posts": [p.model_dump() for p in post_results[:5]],
            "status": "success",
        })

        return GetPostsNeedingContentOutput(
            posts=post_results,
            count=len(post_results),
            status="success",
        )


async def get_posts_needing_embeddings(
    ctx,
    params: GetPostsNeedingEmbeddingsInput,
) -> GetPostsNeedingEmbeddingsOutput:
    """
    Get posts that have content but no embedding yet.
    """
    publication_handles = params.publication_handles
    limit = params.limit

    ctx.report_input({
        "publication_handles": publication_handles,
        "limit": limit,
    })

    async with get_db_session() as db:
        conditions = [
            Post.embedding.is_(None),
            Post.content_raw.isnot(None),
        ]

        if publication_handles:
            conditions.append(Post.publication_handle.in_(publication_handles))

        result = await db.execute(
            select(Post.id, Post.url, Post.title, Post.content_raw)
            .where(and_(*conditions))
            .limit(limit)
        )

        posts = [
            PostNeedingEmbedding(
                id=str(row[0]),
                url=row[1],
                title=row[2] or "",
                content_raw=row[3],
            )
            for row in result.fetchall()
        ]

        logger.info("posts_needing_embeddings_found", count=len(posts))

        ctx.report_output({
            "count": len(posts),
            "sample_posts": [{"id": p.id, "title": p.title[:50]} for p in posts[:5]],
            "status": "success",
        })

        return GetPostsNeedingEmbeddingsOutput(
            posts=posts,
            count=len(posts),
            status="success",
        )


async def update_post_metrics(
    ctx,
    params: UpdatePostMetricsInput,
) -> UpdatePostMetricsOutput:
    """
    Update engagement metrics for posts.
    """
    updates = params.updates

    ctx.report_input({
        "updates_count": len(updates),
    })

    if not updates:
        ctx.report_output({
            "updated_count": 0,
            "status": "success",
        })
        return UpdatePostMetricsOutput(
            updated_count=0,
            status="success",
        )

    async with get_db_session() as db:
        updated_count = 0

        for update in updates:
            url = update.get("url")
            if not url:
                continue

            result = await db.execute(
                select(Post).where(Post.url == url)
            )
            post = result.scalar_one_or_none()

            if post:
                if "likes_count" in update:
                    post.likes_count = update["likes_count"]
                if "comments_count" in update:
                    post.comments_count = update["comments_count"]
                if "shares_count" in update:
                    post.shares_count = update["shares_count"]
                post.metrics_updated_at = datetime.now(timezone.utc)
                updated_count += 1

        await db.commit()

        logger.info("post_metrics_updated", count=updated_count)

        ctx.report_output({
            "updated_count": updated_count,
            "status": "success",
        })

        return UpdatePostMetricsOutput(
            updated_count=updated_count,
            status="success",
        )


async def update_post_embeddings(
    ctx,
    params: UpdatePostEmbeddingsInput,
) -> UpdatePostEmbeddingsOutput:
    """
    Batch update embeddings for posts.
    """
    items = params.items

    ctx.report_input({
        "items_count": len(items),
    })

    if not items:
        ctx.report_output({
            "updated_count": 0,
            "status": "success",
        })
        return UpdatePostEmbeddingsOutput(
            updated_count=0,
            status="success",
        )

    async with get_db_session() as db:
        updated_count = 0

        for item in items:
            item_id = item.get("id")
            embedding = item.get("embedding")

            if not item_id or not embedding:
                continue

            try:
                item_uuid = UUID(item_id) if isinstance(item_id, str) else item_id

                result = await db.execute(
                    select(Post).where(Post.id == item_uuid)
                )
                post = result.scalar_one_or_none()

                if post:
                    post.embedding = embedding
                    updated_count += 1

            except Exception as e:
                logger.warning("post_embedding_update_failed", item_id=item_id, error=str(e))
                continue

        await db.commit()

        logger.info("post_embeddings_updated", count=updated_count)

        ctx.report_output({
            "updated_count": updated_count,
            "status": "success",
        })

        return UpdatePostEmbeddingsOutput(
            updated_count=updated_count,
            status="success",
        )


async def queue_sources_from_posts(
    ctx,
    params: QueueSourcesFromPostsInput,
) -> QueueSourcesFromPostsOutput:
    """
    Auto-pick and queue articles from the unified posts table for mirroring.

    Picks a mix of recent (last N days) and top-performing articles,
    alternating between pools until we reach the requested count.
    """
    source_handles = params.source_handles
    target_handle = params.target_handle
    count = params.count
    min_likes = params.min_likes
    max_age_days = params.max_age_days
    recent_days = params.recent_days

    # Validate
    if not source_handles:
        ctx.report_output({
            "queued_count": 0,
            "status": "error",
            "error": "source_handles is required",
        })
        return QueueSourcesFromPostsOutput(
            queued_count=0,
            status="error",
        )

    if not target_handle:
        ctx.report_output({
            "queued_count": 0,
            "status": "error",
            "error": "target_handle is required",
        })
        return QueueSourcesFromPostsOutput(
            queued_count=0,
            status="error",
        )

    ctx.report_input({
        "source_handles": source_handles,
        "target_handle": target_handle,
        "count": count,
        "min_likes": min_likes,
        "max_age_days": max_age_days,
        "recent_days": recent_days,
    })

    now = datetime.now(timezone.utc)
    max_age_cutoff = now - timedelta(days=max_age_days)
    recent_cutoff = now - timedelta(days=recent_days)

    async with get_db_session() as db:
        # Get post IDs already in queue for this target
        queued_ids_result = await db.execute(
            select(ProductionQueueItem.source_ref_id).where(
                and_(
                    ProductionQueueItem.target_publication == target_handle,
                    ProductionQueueItem.source_ref_id != None,
                )
            )
        )
        queued_ids = set(row[0] for row in queued_ids_result.fetchall())

        # Get post IDs already published for this target
        published_ids_result = await db.execute(
            select(TargetItem.source_ref_id).where(
                and_(
                    TargetItem.target_publication == target_handle,
                    TargetItem.source_ref_id != None,
                )
            )
        )
        published_ids = set(row[0] for row in published_ids_result.fetchall())

        already_processed_ids = queued_ids | published_ids

        logger.info(
            "queue_sources_from_posts_exclusions",
            target=target_handle,
            source_handles=source_handles,
            already_queued=len(queued_ids),
            already_published=len(published_ids),
        )

        # Base conditions for eligible posts
        base_conditions = [
            Post.publication_handle.in_(source_handles),
            Post.has_full_content == True,
            Post.likes_count >= min_likes,
            Post.published_at >= max_age_cutoff,
            Post.content_raw.isnot(None),
        ]
        if already_processed_ids:
            base_conditions.append(~Post.id.in_(already_processed_ids))

        # Get recent pool (last N days, ordered by likes)
        recent_result = await db.execute(
            select(Post).where(
                and_(
                    *base_conditions,
                    Post.published_at >= recent_cutoff,
                )
            ).order_by(Post.likes_count.desc())
        )
        recent_pool = list(recent_result.scalars().all())

        # Get top pool (older than N days, ordered by likes)
        top_result = await db.execute(
            select(Post).where(
                and_(
                    *base_conditions,
                    Post.published_at < recent_cutoff,
                )
            ).order_by(Post.likes_count.desc())
        )
        top_pool = list(top_result.scalars().all())

        logger.info(
            "queue_sources_from_posts_pools",
            recent_pool_size=len(recent_pool),
            top_pool_size=len(top_pool),
        )

        # Alternate between recent and top pools
        selected = []
        recent_idx = 0
        top_idx = 0
        pick_from_recent = True

        while len(selected) < count:
            if pick_from_recent:
                if recent_idx < len(recent_pool):
                    selected.append(("recent", recent_pool[recent_idx]))
                    recent_idx += 1
                elif top_idx < len(top_pool):
                    selected.append(("top", top_pool[top_idx]))
                    top_idx += 1
                else:
                    break
            else:
                if top_idx < len(top_pool):
                    selected.append(("top", top_pool[top_idx]))
                    top_idx += 1
                elif recent_idx < len(recent_pool):
                    selected.append(("recent", recent_pool[recent_idx]))
                    recent_idx += 1
                else:
                    break

            pick_from_recent = not pick_from_recent

        # Queue the selected items
        queued_items = []
        queued_from_recent = 0
        queued_from_top = 0

        # Find max likes for normalization (0-100 scale)
        max_likes_val = max(
            (post.likes_count or 0 for _, post in selected),
            default=1
        ) or 1

        for pool_name, post in selected:
            # Normalize priority to 0-100 scale
            normalized_priority = round((post.likes_count or 0) / max_likes_val * 100, 1)

            queue_item = ProductionQueueItem(
                target_publication=target_handle,
                source_ref_id=post.id,
                title=post.title,
                topic_type="source_derived",
                priority_score=normalized_priority,
                status=QueueStatus.PENDING,
            )
            db.add(queue_item)
            await db.flush()

            queued_items.append(QueuedSourceItem(
                queue_id=str(queue_item.id),
                source_id=str(post.id),
                title=post.title,
            ))

            if pool_name == "recent":
                queued_from_recent += 1
            else:
                queued_from_top += 1

        await db.commit()

        logger.info(
            "queue_sources_from_posts_complete",
            target=target_handle,
            queued=len(queued_items),
            from_recent=queued_from_recent,
            from_top=queued_from_top,
        )

        ctx.report_output({
            "queued_count": len(queued_items),
            "queued_from_recent": queued_from_recent,
            "queued_from_top": queued_from_top,
            "recent_pool_size": len(recent_pool),
            "top_pool_size": len(top_pool),
            "queued_items": [q.model_dump() for q in queued_items[:10]],
            "status": "success" if queued_items else "no_items",
        })

        return QueueSourcesFromPostsOutput(
            queued_count=len(queued_items),
            queued_from_recent=queued_from_recent,
            queued_from_top=queued_from_top,
            queued_items=queued_items,
            status="success" if queued_items else "no_items",
        )


# =============================================================================
# TARGET ITEMS OPERATIONS
# =============================================================================

async def insert_target_items(
    ctx,
    params: InsertTargetItemsInput,
) -> InsertTargetItemsOutput:
    """
    Insert target items (our published content).
    """
    items = params.items
    target_handle = params.target_handle

    ctx.report_input({
        "items_count": len(items),
        "target_handle": target_handle,
    })

    if not items:
        ctx.report_output({
            "inserted_count": 0,
            "status": "success",
        })
        return InsertTargetItemsOutput(
            inserted_count=0,
            status="success",
        )

    async with get_db_session() as db:
        inserted_count = 0

        for item in items:
            # Check if already exists
            result = await db.execute(
                select(TargetItem).where(TargetItem.url == item.url)
            )
            if result.scalar_one_or_none():
                continue

            # Parse published_at
            published_at = item.published_at
            if isinstance(published_at, str):
                published_at = datetime.fromisoformat(published_at.replace("Z", "+00:00"))

            target_item = TargetItem(
                target_publication=target_handle,
                url=item.url,
                title=item.title,
                content_raw=item.content_raw,
                published_at=published_at or datetime.now(timezone.utc),
            )
            db.add(target_item)
            inserted_count += 1

        await db.commit()

        logger.info("target_items_inserted", count=inserted_count)

        ctx.report_output({
            "inserted_count": inserted_count,
            "status": "success",
        })

        return InsertTargetItemsOutput(
            inserted_count=inserted_count,
            status="success",
        )


# =============================================================================
# IDEA GENERATION OPERATIONS
# =============================================================================

async def resolve_source_publications(
    ctx,
    params: ResolveSourcePublicationsInput,
) -> ResolveSourcePublicationsOutput:
    """
    Resolve source publications from KV store or direct input.

    Priority:
    1. If discovery_kv_key provided, fetch from KV store (FAILS if key doesn't exist)
    2. If publications list provided directly, use it
    3. If neither, return no_sources status
    """
    publications = params.publications or []
    discovery_kv_key = params.discovery_kv_key

    ctx.report_input({
        "publications_provided": len(publications) if publications else 0,
        "discovery_kv_key": discovery_kv_key,
    })

    # Option 1: Fetch from KV store (takes priority)
    if discovery_kv_key:
        try:
            kv_data = await ctx.kv.get(discovery_kv_key)

            # Key doesn't exist - FAIL with clear error message
            if kv_data is None:
                logger.error("kv_key_not_found", key=discovery_kv_key)
                raise ValueError(f"KV key '{discovery_kv_key}' not found. Check that the key exists in KV Store and was created by publication_discovery workflow.")

            # Key exists but wrong format
            if not isinstance(kv_data, dict):
                logger.error("kv_data_invalid_format", key=discovery_kv_key, type=type(kv_data).__name__)
                raise ValueError(f"KV key '{discovery_kv_key}' has invalid format: expected dict with 'handles' list, got {type(kv_data).__name__}")

            handles = kv_data.get("handles", [])

            # Key exists but no handles
            if not handles:
                logger.error("kv_key_no_handles", key=discovery_kv_key)
                raise ValueError(f"KV key '{discovery_kv_key}' found but 'handles' list is empty. Available keys in data: {list(kv_data.keys())}")

            # Success
            ctx.report_output({
                "source": "kv",
                "kv_key": discovery_kv_key,
                "publications_count": len(handles),
                "publications": handles,
                "kv_metadata": {
                    "category": kv_data.get("category"),
                    "keywords": kv_data.get("keywords"),
                    "generated_at": kv_data.get("generated_at"),
                },
                "status": "success",
            })
            return ResolveSourcePublicationsOutput(
                publications=handles,
                source="kv",
                count=len(handles),
                status="success",
            )
        except Exception as e:
            logger.error("kv_fetch_failed", key=discovery_kv_key, error=str(e))
            ctx.report_output({
                "source": "kv",
                "kv_key": discovery_kv_key,
                "error": str(e),
                "status": "error",
            })
            return ResolveSourcePublicationsOutput(
                publications=[],
                source="kv",
                count=0,
                status="error",
            )

    # Option 2: Direct publications list
    if publications and len(publications) > 0:
        ctx.report_output({
            "source": "input",
            "publications_count": len(publications),
            "publications": publications,
            "status": "success",
        })
        return ResolveSourcePublicationsOutput(
            publications=publications,
            source="input",
            count=len(publications),
            status="success",
        )

    # Neither provided
    ctx.report_output({
        "error": "No publications provided and no discovery_kv_key specified",
        "status": "no_sources",
    })
    return ResolveSourcePublicationsOutput(
        publications=[],
        source="",
        count=0,
        status="no_sources",
    )


async def fetch_top_source_posts(
    ctx,
    params: FetchTopSourcePostsInput,
) -> FetchTopSourcePostsOutput:
    """
    Fetch top-performing posts from source publications.

    Gets the top N posts (by likes) from each source publication,
    filtered to posts from the last max_age_days.

    If target_handle is provided, also filters out posts that have already
    been used to generate ideas for that target (stored in idea_source_posts).
    """
    publications = params.publications or []
    posts_per_source = params.posts_per_source
    max_age_days = params.max_age_days
    target_handle = params.target_handle

    ctx.report_input({
        "publications": publications,
        "publications_count": len(publications),
        "posts_per_source": posts_per_source,
        "max_age_days": max_age_days,
        "target_handle": target_handle,
    })

    if not publications:
        ctx.report_output({
            "error": "No publications provided",
            "status": "error",
        })
        return FetchTopSourcePostsOutput(
            source_posts=[],
            total_count=0,
            sources_count=0,
            status="error",
        )

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    async with get_db_session() as db:
        # Get IDs of posts already used for this target (if target_handle provided)
        worked_post_ids = set()
        if target_handle:
            worked_result = await db.execute(
                select(IdeaSourcePost.source_post_id)
                .where(IdeaSourcePost.target_handle == target_handle)
                # Future: add decay filter here, e.g.:
                # .where(IdeaSourcePost.created_at > datetime.now(timezone.utc) - timedelta(days=60))
            )
            worked_post_ids = {row[0] for row in worked_result.fetchall()}

        all_posts = []
        filtered_count = 0
        sources_with_posts = 0

        for handle in publications:
            # Get top posts by likes for this publication
            query = select(Post).where(
                and_(
                    Post.publication_handle == handle,
                    Post.published_at >= cutoff_date,
                )
            )

            # Exclude already-worked posts if we have a target
            if worked_post_ids:
                query = query.where(Post.id.notin_(worked_post_ids))

            result = await db.execute(
                query.order_by(Post.likes_count.desc()).limit(posts_per_source)
            )
            posts = result.scalars().all()

            if posts:
                sources_with_posts += 1
                for p in posts:
                    all_posts.append(SourcePost(
                        id=str(p.id),
                        title=p.title,
                        publication_handle=p.publication_handle,
                        likes_count=p.likes_count or 0,
                        # Additional context for better LLM idea generation
                        subtitle=p.subtitle,
                        tags=p.tags or [],
                        section_name=p.section_name,
                        word_count=p.word_count,
                        comments_count=p.comments_count or 0,
                    ))

        # Count how many posts were filtered out
        if target_handle and worked_post_ids:
            filtered_count = len(worked_post_ids)

        logger.info(
            "top_source_posts_fetched",
            total_posts=len(all_posts),
            sources_with_posts=sources_with_posts,
            sources_requested=len(publications),
            worked_posts_filtered=filtered_count,
        )

        ctx.report_output({
            "total_posts": len(all_posts),
            "sources_with_posts": sources_with_posts,
            "sources_without_posts": len(publications) - sources_with_posts,
            "worked_posts_filtered": filtered_count,
            "sample_posts": [p.model_dump() for p in all_posts[:10]],
            "status": "success",
        })

        return FetchTopSourcePostsOutput(
            source_posts=[p.model_dump() for p in all_posts],
            total_count=len(all_posts),
            sources_count=sources_with_posts,
            status="success",
        )


async def filter_worked_source_posts(
    ctx,
    params: FilterWorkedSourcePostsInput,
) -> FilterWorkedSourcePostsOutput:
    """
    Filter out source posts that have already been worked on.

    Removes posts that are:
    - Already in the production queue (any status)
    - Already published as target items
    """
    target_handle = params.target_handle
    posts = params.posts

    ctx.report_input({
        "target_handle": target_handle,
        "posts_count": len(posts),
    })

    if not posts:
        ctx.report_output({
            "filtered_count": 0,
            "removed_count": 0,
            "status": "success",
        })
        return FilterWorkedSourcePostsOutput(
            source_posts=[],
            filtered_count=0,
            removed_count=0,
            status="success",
        )

    post_ids = [p.get("id") for p in posts if p.get("id")]

    async with get_db_session() as db:
        # Get IDs of posts already in queue
        queue_result = await db.execute(
            select(ProductionQueueItem.source_ref_id)
            .where(
                and_(
                    ProductionQueueItem.target_publication == target_handle,
                    ProductionQueueItem.source_ref_id.isnot(None),
                )
            )
        )
        queued_ids = {str(row[0]) for row in queue_result.fetchall() if row[0]}

        # Get IDs of posts already published
        published_result = await db.execute(
            select(TargetItem.source_ref_id)
            .where(
                and_(
                    TargetItem.target_publication == target_handle,
                    TargetItem.source_ref_id.isnot(None),
                )
            )
        )
        published_ids = {str(row[0]) for row in published_result.fetchall() if row[0]}

        worked_ids = queued_ids | published_ids

        # Filter out worked posts
        filtered_posts = [p for p in posts if p.get("id") not in worked_ids]
        removed_count = len(posts) - len(filtered_posts)

        logger.info(
            "source_posts_filtered",
            original_count=len(posts),
            filtered_count=len(filtered_posts),
            removed_count=removed_count,
            queued_ids_count=len(queued_ids),
            published_ids_count=len(published_ids),
        )

        ctx.report_output({
            "original_count": len(posts),
            "filtered_count": len(filtered_posts),
            "removed_count": removed_count,
            "removed_reasons": {
                "in_queue": len(queued_ids & set(post_ids)),
                "already_published": len(published_ids & set(post_ids)),
            },
            "status": "success",
        })

        return FilterWorkedSourcePostsOutput(
            source_posts=filtered_posts,
            filtered_count=len(filtered_posts),
            removed_count=removed_count,
            status="success",
        )


async def fetch_my_published_posts(
    ctx,
    params: FetchMyPublishedPostsInput,
) -> FetchMyPublishedPostsOutput:
    """
    Fetch published posts for a target publication from the posts table.
    Used for voice/style context in idea generation.
    """
    target_handle = params.target_handle
    limit = params.limit
    content_preview_length = params.content_preview_length

    ctx.report_input({
        "target_handle": target_handle,
        "limit": limit,
        "content_preview_length": content_preview_length,
    })

    async with get_db_session() as db:
        # Query the posts table (where content_sync stores scraped posts)
        result = await db.execute(
            select(Post)
            .where(Post.publication_handle == target_handle)
            .order_by(Post.published_at.desc())
            .limit(limit)
        )
        items = result.scalars().all()

        posts = [
            MyPublishedPost(
                id=str(item.id),
                url=item.url,
                title=item.title,
                content_preview=(item.content_raw[:content_preview_length] + "...") if item.content_raw and len(item.content_raw) > content_preview_length else (item.content_raw or ""),
                likes_count=item.likes_count or 0,
                published_at=item.published_at.isoformat() if item.published_at else None,
            )
            for item in items
        ]

        logger.info("my_published_posts_fetched", handle=target_handle, count=len(posts))

        ctx.report_output({
            "posts_count": len(posts),
            "sample_posts": [p.model_dump() for p in posts[:5]],
            "status": "success",
        })

        return FetchMyPublishedPostsOutput(
            my_posts=[p.model_dump() for p in posts],
            count=len(posts),
            status="success",
        )


# =============================================================================
# PRODUCTION QUEUE OPERATIONS
# =============================================================================

async def queue_gap_ideas(
    ctx,
    params: QueueGapIdeasInput,
) -> QueueGapIdeasOutput:
    """
    Queue gap analysis ideas (original topic ideas).
    """
    ideas = params.ideas
    target_handle = params.target_handle

    ctx.report_input({
        "ideas_count": len(ideas),
        "target_handle": target_handle,
    })

    if not ideas:
        ctx.report_output({
            "queued_count": 0,
            "status": "success",
        })
        return QueueGapIdeasOutput(
            queued_count=0,
            queued_ideas=[],
            status="success",
        )

    async with get_db_session() as db:
        queued_ideas = []

        # Collect all source post IDs to fetch likes in one query
        all_source_ids = set()
        for idea in ideas:
            if isinstance(idea, dict):
                source_ids = idea.get("source_post_ids", [])
            else:
                source_ids = getattr(idea, "source_post_ids", [])
            for sid in source_ids:
                try:
                    all_source_ids.add(UUID(sid) if isinstance(sid, str) else sid)
                except (ValueError, AttributeError):
                    pass

        # Fetch likes for all source posts in one query
        source_likes_map = {}  # UUID -> likes_count
        if all_source_ids:
            result = await db.execute(
                select(Post.id, Post.likes_count).where(Post.id.in_(list(all_source_ids)))
            )
            for post_id, likes in result.fetchall():
                source_likes_map[post_id] = likes or 0

        def calculate_priority_score(source_post_ids: list, post_type: str) -> float:
            """Calculate priority score based on source post engagement.

            Score formula:
            - Base: 30 (minimum for any idea)
            - Likes contribution: up to 50 points based on log scale of average likes
            - Paid bonus: +10 for paid content (more valuable/exclusive)

            Scale: 100 likes = ~50, 1000 likes = ~70, 10000 likes = ~90
            """
            import math

            if not source_post_ids:
                # Original idea with no source - base score
                return 40.0 if post_type == "paid" else 30.0

            # Get likes for source posts
            likes_values = []
            for sid in source_post_ids:
                try:
                    post_uuid = UUID(sid) if isinstance(sid, str) else sid
                    if post_uuid in source_likes_map:
                        likes_values.append(source_likes_map[post_uuid])
                except (ValueError, AttributeError):
                    pass

            if not likes_values:
                return 40.0 if post_type == "paid" else 30.0

            avg_likes = sum(likes_values) / len(likes_values)

            # Log scale: log10(likes+1) * 20, capped at 50
            # 10 likes = 20, 100 likes = 40, 1000 likes = 60 (capped at 50)
            likes_score = min(50.0, math.log10(avg_likes + 1) * 20)

            # Base + likes + paid bonus
            base = 30.0
            paid_bonus = 10.0 if post_type == "paid" else 0.0

            return min(100.0, base + likes_score + paid_bonus)

        for idea in ideas:
            # Handle both dict and object cases
            if isinstance(idea, dict):
                title = idea.get("title", "")
                # post_type from LLM output maps to topic_type in queue
                post_type = idea.get("post_type", "free")
                topic_type = f"idea_{post_type}" if post_type else "idea_free"
                summary = idea.get("summary", "")
                rationale = idea.get("rationale", "")
                inspiration = idea.get("inspiration_source", "")
                source_post_ids = idea.get("source_post_ids", [])
            else:
                title = idea.title
                post_type = getattr(idea, "post_type", "free")
                topic_type = getattr(idea, "topic_type", None) or (f"idea_{post_type}" if post_type else "idea_free")
                summary = getattr(idea, "summary", "")
                rationale = getattr(idea, "rationale", "")
                inspiration = getattr(idea, "inspiration_source", "")
                source_post_ids = getattr(idea, "source_post_ids", [])

            # Calculate priority score based on source post likes
            priority_score = calculate_priority_score(source_post_ids, post_type)

            queue_item = ProductionQueueItem(
                target_publication=target_handle,
                title=title,
                topic_type=topic_type,
                priority_score=priority_score,
                status=QueueStatus.PENDING,
                draft_metadata={
                    "summary": summary,
                    "rationale": rationale,
                    "inspiration_source": inspiration,
                    "post_type": post_type,  # "paid" or "free"
                    "source_post_ids": source_post_ids,  # Track which posts inspired this
                },
            )
            db.add(queue_item)
            await db.flush()

            # Record which source posts inspired this idea (for filtering in future runs)
            # Deduplicate source_post_ids to avoid unique constraint violations
            seen_source_ids = set()
            for source_id in source_post_ids:
                if source_id in seen_source_ids:
                    continue
                seen_source_ids.add(source_id)
                try:
                    idea_source = IdeaSourcePost(
                        queue_item_id=queue_item.id,
                        source_post_id=UUID(source_id) if isinstance(source_id, str) else source_id,
                        target_handle=target_handle,
                    )
                    db.add(idea_source)
                except Exception as e:
                    logger.warning("failed_to_record_source_post", source_id=source_id, error=str(e))

            queued_ideas.append(QueuedIdea(
                queue_id=str(queue_item.id),
                title=title,
            ))

        await db.commit()

        logger.info("gap_ideas_queued", count=len(queued_ideas))

        ctx.report_output({
            "queued_count": len(queued_ideas),
            "queued_ideas": [q.model_dump() for q in queued_ideas],
            "status": "success",
        })

        return QueueGapIdeasOutput(
            queued_count=len(queued_ideas),
            queued_ideas=queued_ideas,
            status="success",
        )


async def get_existing_topics(
    ctx,
    params: GetExistingTopicsInput,
) -> GetExistingTopicsOutput:
    """
    Get existing topics from queue and published items (for dedup).
    """
    target_handle = params.target_handle

    ctx.report_input({
        "target_handle": target_handle,
    })

    async with get_db_session() as db:
        # Get queue titles
        queue_result = await db.execute(
            select(ProductionQueueItem.title)
            .where(ProductionQueueItem.target_publication == target_handle)
        )
        queue_titles = [row[0] for row in queue_result.fetchall()]

        # Get published titles
        published_result = await db.execute(
            select(TargetItem.title)
            .where(TargetItem.target_publication == target_handle)
        )
        published_titles = [row[0] for row in published_result.fetchall()]

        all_topics = list(set(queue_titles + published_titles))

        logger.info("existing_topics_fetched", count=len(all_topics))

        ctx.report_output({
            "topics_count": len(all_topics),
            "status": "success",
        })

        return GetExistingTopicsOutput(
            topics=all_topics,
            count=len(all_topics),
            status="success",
        )


async def get_pending_queue_items(
    ctx,
    params: GetPendingQueueItemsInput,
) -> GetPendingQueueItemsOutput:
    """
    Get pending queue items for drafting.

    If queue_id is provided, fetches that specific item (for re-runs).
    Otherwise, fetches next pending items by priority.
    """
    target_handle = params.target_handle
    limit = params.limit
    queue_id = params.queue_id

    ctx.report_input({
        "target_handle": target_handle,
        "limit": limit,
        "queue_id": queue_id,
    })

    async with get_db_session() as db:
        if queue_id:
            # Fetch specific item by ID (for re-runs)
            result = await db.execute(
                select(ProductionQueueItem)
                .where(ProductionQueueItem.id == queue_id)
            )
            items = result.scalars().all()
        else:
            # Fetch next pending items by priority
            result = await db.execute(
                select(ProductionQueueItem)
                .where(
                    and_(
                        ProductionQueueItem.target_publication == target_handle,
                        ProductionQueueItem.status == QueueStatus.PENDING,
                    )
                )
                .order_by(ProductionQueueItem.priority_score.desc())
                .limit(limit)
            )
            items = result.scalars().all()

        pending_items = []
        for item in items:
            # Lock item by setting status to drafting
            item.status = QueueStatus.DRAFTING

            # Extract idea context from draft_metadata (populated by queue_gap_ideas)
            metadata = item.draft_metadata or {}
            pending_items.append(PendingQueueItem(
                id=str(item.id),
                title=item.title,
                topic_type=item.topic_type,
                priority_score=item.priority_score or 0,
                source_ref_id=str(item.source_ref_id) if item.source_ref_id else None,
                # Rich context from idea generation
                idea_summary=metadata.get("summary"),
                idea_inspiration=metadata.get("inspiration_source"),
                post_type=metadata.get("post_type") or (item.topic_type.replace("idea_", "") if item.topic_type and item.topic_type.startswith("idea_") else None),
            ))

        # Commit the status change to lock the items
        await db.commit()

        logger.info("pending_queue_items_fetched", count=len(pending_items))

        # Determine status for routing
        if len(pending_items) == 0:
            status = "no_pending_items"
        else:
            status = "success"

        ctx.report_output({
            "items_count": len(pending_items),
            "items": [p.model_dump() for p in pending_items],
            "status": status,
        })

        return GetPendingQueueItemsOutput(
            items=pending_items,
            count=len(pending_items),
            status=status,
        )


async def load_drafting_context(
    ctx,
    params: LoadDraftingContextInput,
) -> LoadDraftingContextOutput:
    """
    Load context for drafting: fetch source content (if source_ref_id provided)
    and calculate target word count based on length_mode.
    """
    source_ref_id = params.source_ref_id
    length_mode = params.length_mode or "standard"
    custom_word_count = params.custom_word_count

    ctx.report_input({
        "source_ref_id": source_ref_id,
        "length_mode": length_mode,
        "custom_word_count": custom_word_count,
    })

    # Calculate target word count based on length_mode
    word_count_map = {
        "brief": 700,
        "standard": 1250,
        "long": 2500,
    }
    if length_mode == "custom" and custom_word_count:
        target_word_count = custom_word_count
    else:
        target_word_count = word_count_map.get(length_mode, 1250)

    # Get source post if source_ref_id provided
    source_content = None

    if source_ref_id:
        async with get_db_session() as db:
            try:
                source_result = await db.execute(
                    select(Post).where(Post.id == UUID(source_ref_id))
                )
                source_post = source_result.scalar_one_or_none()

                if source_post:
                    source_content = source_post.content_raw
            except Exception as e:
                logger.warning(f"Failed to load source post {source_ref_id}: {e}")

    ctx.report_output({
        "has_source_content": bool(source_content),
        "target_word_count": target_word_count,
        "status": "success",
    })

    return LoadDraftingContextOutput(
        source_content=source_content,
        target_word_count=target_word_count,
        status="success",
    )


async def update_queue_item_draft(
    ctx,
    params: UpdateQueueItemDraftInput,
) -> UpdateQueueItemDraftOutput:
    """
    Update queue item with draft content and metadata.
    """
    queue_id = params.queue_id
    draft_content = params.draft_content
    draft_image_url = params.draft_image_url
    draft_metadata = params.draft_metadata

    ctx.report_input({
        "queue_id": queue_id,
        "has_draft_content": bool(draft_content),
        "has_draft_image_url": bool(draft_image_url),
        "has_draft_metadata": bool(draft_metadata),
        "draft_metadata_keys": list(draft_metadata.keys()) if draft_metadata else [],
        "draft_metadata_preview": {k: str(v)[:100] if v else None for k, v in (draft_metadata or {}).items()},
    })

    async with get_db_session() as db:
        result = await db.execute(
            select(ProductionQueueItem).where(ProductionQueueItem.id == UUID(queue_id))
        )
        queue_item = result.scalar_one_or_none()

        if not queue_item:
            ctx.report_output({"status": "not_found"})
            return UpdateQueueItemDraftOutput(status="not_found")

        queue_item.draft_content = draft_content
        queue_item.draft_image_url = draft_image_url
        queue_item.status = QueueStatus.REVIEW
        queue_item.drafted_at = datetime.now(timezone.utc)

        # Merge draft_metadata into existing metadata (preserve existing fields)
        if draft_metadata:
            existing_metadata = dict(queue_item.draft_metadata or {})
            logger.info("draft_metadata_merge",
                        existing_keys=list(existing_metadata.keys()),
                        new_keys=list(draft_metadata.keys()))
            existing_metadata.update(draft_metadata)
            queue_item.draft_metadata = existing_metadata
            # Tell SQLAlchemy the JSON field changed (it doesn't detect mutable changes automatically)
            attributes.flag_modified(queue_item, "draft_metadata")
            logger.info("draft_metadata_after_merge",
                        final_keys=list(queue_item.draft_metadata.keys()))

        await db.commit()

        logger.info("queue_item_draft_updated",
                    queue_id=queue_id,
                    final_metadata_keys=list(queue_item.draft_metadata.keys()) if queue_item.draft_metadata else [])

        ctx.report_output({
            "queue_id": queue_id,
            "status": "success",
        })

        return UpdateQueueItemDraftOutput(
            queue_id=queue_id,
            status="success",
        )


async def set_queue_item_drafting(
    ctx,
    params: SetQueueItemDraftingInput,
) -> SetQueueItemDraftingOutput:
    """
    Set queue item status to drafting.
    """
    queue_id = params.queue_id

    ctx.report_input({
        "queue_id": queue_id,
    })

    async with get_db_session() as db:
        result = await db.execute(
            select(ProductionQueueItem).where(ProductionQueueItem.id == UUID(queue_id))
        )
        queue_item = result.scalar_one_or_none()

        if not queue_item:
            ctx.report_output({"status": "not_found"})
            return SetQueueItemDraftingOutput(status="not_found")

        queue_item.status = QueueStatus.DRAFTING

        await db.commit()

        logger.info("queue_item_set_drafting", queue_id=queue_id)

        ctx.report_output({
            "queue_id": queue_id,
            "status": "success",
        })

        return SetQueueItemDraftingOutput(
            queue_id=queue_id,
            status="success",
        )


async def reset_drafting_status(
    ctx,
    params: ResetDraftingStatusInput,
) -> ResetDraftingStatusOutput:
    """
    Reset a queue item from drafting back to pending.
    """
    queue_id = params.queue_id

    ctx.report_input({
        "queue_id": queue_id,
    })

    async with get_db_session() as db:
        result = await db.execute(
            select(ProductionQueueItem).where(ProductionQueueItem.id == UUID(queue_id))
        )
        item = result.scalar_one_or_none()

        if not item:
            ctx.report_output({"status": "not_found"})
            return ResetDraftingStatusOutput(
                queue_id=queue_id,
                reset=False,
                status="not_found",
            )

        if item.status == QueueStatus.DRAFTING:
            item.status = QueueStatus.PENDING
            await db.commit()

            logger.info("drafting_status_reset", queue_id=queue_id)

            ctx.report_output({
                "queue_id": queue_id,
                "reset": True,
                "status": "success",
            })

            return ResetDraftingStatusOutput(
                queue_id=queue_id,
                reset=True,
                status="success",
            )
        else:
            ctx.report_output({
                "queue_id": queue_id,
                "reset": False,
                "current_status": item.status.value,
                "status": "not_drafting",
            })

            return ResetDraftingStatusOutput(
                queue_id=queue_id,
                reset=False,
                status="not_drafting",
            )


async def reset_stale_drafting_items(
    ctx,
    params: ResetStaleDraftingItemsInput,
) -> ResetStaleDraftingItemsOutput:
    """
    Reset queue items stuck in 'drafting' status for too long.
    """
    target_handle = params.target_handle
    stale_minutes = params.stale_minutes

    ctx.report_input({
        "target_handle": target_handle,
        "stale_minutes": stale_minutes,
    })

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=stale_minutes)

    async with get_db_session() as db:
        result = await db.execute(
            select(ProductionQueueItem)
            .where(
                and_(
                    ProductionQueueItem.target_publication == target_handle,
                    ProductionQueueItem.status == QueueStatus.DRAFTING,
                    ProductionQueueItem.updated_at < cutoff,
                )
            )
        )
        stale_items = result.scalars().all()

        reset_ids = []
        for item in stale_items:
            item.status = QueueStatus.PENDING
            reset_ids.append(str(item.id))

        if reset_ids:
            await db.commit()
            logger.warning(
                "stale_drafting_items_reset",
                target_handle=target_handle,
                count=len(reset_ids),
                stale_minutes=stale_minutes,
            )

        ctx.report_output({
            "reset_count": len(reset_ids),
            "reset_ids": reset_ids,
            "status": "success",
        })

        return ResetStaleDraftingItemsOutput(
            reset_count=len(reset_ids),
            reset_ids=reset_ids,
            status="success",
        )


async def get_queue_item_for_publish(
    ctx,
    params: GetQueueItemForPublishInput,
) -> GetQueueItemForPublishOutput:
    """
    Get queue item ready for publishing.
    """
    queue_id = params.queue_id

    ctx.report_input({"queue_id": queue_id})

    async with get_db_session() as db:
        result = await db.execute(
            select(ProductionQueueItem).where(ProductionQueueItem.id == UUID(queue_id))
        )
        item = result.scalar_one_or_none()

        if not item:
            ctx.report_output({"status": "not_found"})
            return GetQueueItemForPublishOutput(status="not_found")

        if item.status != QueueStatus.REVIEW:
            ctx.report_output({"status": "not_ready", "current_status": item.status.value})
            return GetQueueItemForPublishOutput(status="not_ready")

        queue_item = QueueItemForPublish(
            id=str(item.id),
            title=item.title,
            draft_content=item.draft_content,
            draft_image_url=item.draft_image_url,
        )

        ctx.report_output({
            "item": queue_item.model_dump(),
            "status": "success",
        })

        return GetQueueItemForPublishOutput(
            item=queue_item,
            status="success",
        )


async def finalize_publication(
    ctx,
    params: FinalizePublicationInput,
) -> FinalizePublicationOutput:
    """
    Finalize publication: update queue and create target item.
    """
    target_handle = params.target_handle
    queue_id = params.queue_id
    published_url = params.published_url
    published_title = params.published_title

    ctx.report_input({
        "target_handle": target_handle,
        "queue_id": queue_id,
        "published_url": published_url,
        "published_title": published_title,
    })

    async with get_db_session() as db:
        result = await db.execute(
            select(ProductionQueueItem).where(ProductionQueueItem.id == UUID(queue_id))
        )
        queue_item = result.scalar_one_or_none()

        if not queue_item:
            ctx.report_output({"status": "not_found"})
            return FinalizePublicationOutput(status="not_found")

        # Update queue status
        queue_item.status = QueueStatus.PUBLISHED
        queue_item.published_at = datetime.now(timezone.utc)

        # Create target_item
        target_item = TargetItem(
            target_publication=target_handle,
            source_ref_id=queue_item.source_ref_id,
            url=published_url,
            title=published_title,
            content_raw=queue_item.draft_content,
            embedding=queue_item.embedding,
            published_at=datetime.now(timezone.utc),
        )
        db.add(target_item)
        await db.flush()

        await db.commit()

        logger.info(
            "publication_finalized",
            queue_id=queue_id,
            target_id=str(target_item.id),
            url=published_url,
        )

        ctx.report_output({
            "target_item_id": str(target_item.id),
            "published_url": published_url,
            "status": "success",
        })

        return FinalizePublicationOutput(
            target_item_id=str(target_item.id),
            published_url=published_url,
            status="success",
        )


# =============================================================================
# TARGET CONFIG OPERATIONS
# =============================================================================

async def load_target_config(
    ctx,
    params: LoadTargetConfigInput,
) -> LoadTargetConfigOutput:
    """
    Load target config (prompts, cookies, paid subscription whitelist).

    Priority order for prompts:
    1. Vault configs (via ctx.get_config() - bound in deployment)
    2. Legacy target_config table (fallback for backwards compatibility)
    """
    target_handle = params.target_handle

    ctx.report_input({
        "target_handle": target_handle,
    })

    # Priority 1: Vault configs (via ctx.get_config)
    # These are bound in the deployment's vault_bindings.configs
    article_prompt = ""
    hero_image_prompt = ""
    inline_image_prompt = ""

    config = ctx.get_config("article_prompt")
    if config:
        # Config value is usually {"text": "..."} or just the text
        article_prompt = config.get("text", config) if isinstance(config, dict) else str(config)

    config = ctx.get_config("hero_image_style")
    if config:
        hero_image_prompt = config.get("text", config) if isinstance(config, dict) else str(config)

    config = ctx.get_config("inline_image_style")
    if config:
        inline_image_prompt = config.get("text", config) if isinstance(config, dict) else str(config)

    # Track sources for debugging
    article_source = "vault_config" if article_prompt else "none"
    hero_source = "vault_config" if hero_image_prompt else "none"
    inline_source = "vault_config" if inline_image_prompt else "none"

    # Priority 2: Legacy target_config table (fallback)
    substack_cookies = None
    paid_subscription_handles = []

    async with get_db_session() as db:
        result = await db.execute(
            text("""
                SELECT article_prompt, hero_image_prompt, inline_image_prompt,
                       substack_cookies, paid_subscription_handles
                FROM target_config
                WHERE target_handle = :target_handle
            """),
            {"target_handle": target_handle}
        )
        row = result.fetchone()

        if row:
            if not article_prompt:
                article_prompt = row[0] or ""
                article_source = "database" if article_prompt else "none"
            if not hero_image_prompt:
                hero_image_prompt = row[1] or ""
                hero_source = "database" if hero_image_prompt else "none"
            if not inline_image_prompt:
                inline_image_prompt = row[2] or ""
                inline_source = "database" if inline_image_prompt else "none"
            substack_cookies = row[3]
            paid_subscription_handles = row[4] or []

    # Build warnings for missing configs
    missing_configs = []
    if not article_prompt:
        missing_configs.append("article_prompt")
        ctx.warning("Missing config: article_prompt - will use generic fallback prompt")
    if not hero_image_prompt:
        missing_configs.append("hero_image_style")
        ctx.warning("Missing config: hero_image_style - hero images will have no style guidance")
    if not inline_image_prompt:
        missing_configs.append("inline_image_style")
        ctx.warning("Missing config: inline_image_style - inline images will have no style guidance")

    ctx.report_output({
        "article_prompt": article_prompt[:200] + "..." if len(article_prompt) > 200 else article_prompt,
        "hero_image_prompt": hero_image_prompt[:200] + "..." if len(hero_image_prompt) > 200 else hero_image_prompt,
        "inline_image_prompt": inline_image_prompt[:200] + "..." if len(inline_image_prompt) > 200 else inline_image_prompt,
        "article_prompt_source": article_source,
        "hero_image_prompt_source": hero_source,
        "inline_image_prompt_source": inline_source,
        "missing_configs": missing_configs,
        "has_substack_cookies": bool(substack_cookies),
        "paid_subscription_handles": paid_subscription_handles,
        "status": "success",
    })

    return LoadTargetConfigOutput(
        article_prompt=article_prompt,
        hero_image_prompt=hero_image_prompt,
        inline_image_prompt=inline_image_prompt,
        substack_cookies=substack_cookies,
        paid_subscription_handles=paid_subscription_handles,
        status="success",
    )
