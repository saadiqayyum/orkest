"""
RAG (Retrieval Augmented Generation) node functions for the AI Substack Mirroring Engine.

These workflow nodes handle:
- Vector similarity search
- Soft deduplication
- Scoring and selection
- Context retrieval for drafting
"""
from datetime import datetime, timezone
import structlog

from sqlalchemy import select

from shared.database import get_db_session
from shared.models import Post, TargetItem, ProductionQueueItem

from .schemas import (
    VectorDedupInput, VectorDedupOutput, IdeaForDedup,
    FindSimilarPostsInput, FindSimilarPostsOutput, SimilarPost,
    GetStyleReferenceInput, GetStyleReferenceOutput, StyleReferencePost,
    CheckTopicUniquenessInput, CheckTopicUniquenessOutput, SimilarItem,
)

logger = structlog.get_logger()

# Default deduplication threshold (cosine distance)
# Lower = more strict (fewer false positives)
# 0.15 = ~85% similarity threshold
DEFAULT_DEDUP_THRESHOLD = 0.15


async def vector_dedup(
    ctx,
    params: VectorDedupInput,
) -> VectorDedupOutput:
    """
    Soft Dedup: Check ideas against existing content using vector similarity.

    For each idea:
    1. Vector search against production_queue AND target_items
    2. If distance < threshold, DISCARD (too similar)
    3. Cap output at max_results if specified
    """
    ideas = params.ideas
    threshold = params.threshold
    max_results = params.max_results

    ctx.report_input({
        "ideas_count": len(ideas),
        "threshold": threshold,
        "max_results": max_results,
    })

    surviving = []
    discarded = []

    async with get_db_session() as db:
        for idea in ideas:
            # Handle both dicts and Pydantic models
            if isinstance(idea, dict):
                embedding = idea.get("embedding")
                topic = idea.get("title", "")
                post_type = idea.get("post_type", "free")
                summary = idea.get("summary", "")
            else:
                embedding = getattr(idea, "embedding", None)
                topic = getattr(idea, "title", "")
                post_type = getattr(idea, "post_type", "free")
                summary = getattr(idea, "summary", "")

            if not embedding:
                logger.warning("idea_missing_embedding", topic=topic[:50])
                discarded.append({
                    "title": topic,
                    "post_type": post_type,
                    "reason": "missing_embedding",
                    "distance": None,
                    "similar_to": None,
                })
                continue

            # Track closest matches for surviving ideas
            closest_queue_distance = None
            closest_queue_title = None
            closest_target_distance = None
            closest_target_title = None

            # Check against production_queue
            queue_similar = await db.execute(
                select(ProductionQueueItem.id, ProductionQueueItem.title)
                .where(ProductionQueueItem.embedding.isnot(None))
                .order_by(ProductionQueueItem.embedding.cosine_distance(embedding))
                .limit(1)
            )
            queue_match = queue_similar.fetchone()

            if queue_match:
                # Calculate actual distance
                distance_result = await db.execute(
                    select(ProductionQueueItem.embedding.cosine_distance(embedding))
                    .where(ProductionQueueItem.id == queue_match[0])
                )
                distance = distance_result.scalar()
                closest_queue_distance = round(distance, 4) if distance else None
                closest_queue_title = queue_match[1][:60] if queue_match[1] else None

                if distance is not None and distance < threshold:
                    logger.info(
                        "idea_discarded_queue_match",
                        topic=topic[:50],
                        similar_to=queue_match[1][:50] if queue_match[1] else "",
                        distance=distance,
                    )
                    discarded.append({
                        "title": topic,
                        "post_type": post_type,
                        "reason": "similar_to_queue",
                        "similar_to": queue_match[1],
                        "distance": round(distance, 4),
                        "threshold": threshold,
                    })
                    continue

            # Check against target_items (published content)
            target_similar = await db.execute(
                select(TargetItem.id, TargetItem.title)
                .where(TargetItem.embedding.isnot(None))
                .order_by(TargetItem.embedding.cosine_distance(embedding))
                .limit(1)
            )
            target_match = target_similar.fetchone()

            if target_match:
                distance_result = await db.execute(
                    select(TargetItem.embedding.cosine_distance(embedding))
                    .where(TargetItem.id == target_match[0])
                )
                distance = distance_result.scalar()
                closest_target_distance = round(distance, 4) if distance else None
                closest_target_title = target_match[1][:60] if target_match[1] else None

                if distance is not None and distance < threshold:
                    logger.info(
                        "idea_discarded_target_match",
                        topic=topic[:50],
                        similar_to=target_match[1][:50] if target_match[1] else "",
                        distance=distance,
                    )
                    discarded.append({
                        "title": topic,
                        "post_type": post_type,
                        "reason": "similar_to_published",
                        "similar_to": target_match[1],
                        "distance": round(distance, 4),
                        "threshold": threshold,
                    })
                    continue

            # Passed both checks - keep the idea with distance info
            idea_dict = idea if isinstance(idea, dict) else (idea.__dict__ if hasattr(idea, '__dict__') else {})
            surviving.append({
                **idea_dict,
                "_closest_queue": {"title": closest_queue_title, "distance": closest_queue_distance},
                "_closest_published": {"title": closest_target_title, "distance": closest_target_distance},
            })

            # Stop early if we've reached max_results
            if max_results and len(surviving) >= max_results:
                logger.info("max_results_reached", max_results=max_results)
                break

    # Cap surviving at max_results (in case loop didn't catch it)
    if max_results and len(surviving) > max_results:
        surviving = surviving[:max_results]

    logger.info(
        "soft_dedup_complete",
        total=len(ideas),
        surviving=len(surviving),
        discarded=len(discarded),
        threshold=threshold,
        max_results=max_results,
    )

    # Report output - show surviving and discarded with full details
    ctx.report_output({
        "threshold": threshold,
        "max_results": max_results,
        "input_count": len(ideas),
        "surviving_count": len(surviving),
        "discarded_count": len(discarded),
        "surviving_ideas": [
            {
                "title": s.get("title", "") if isinstance(s, dict) else getattr(s, "title", ""),
                "post_type": s.get("post_type", "free") if isinstance(s, dict) else getattr(s, "post_type", "free"),
                "summary": (s.get("summary", "")[:100] + "...") if isinstance(s, dict) and s.get("summary") else "",
                "closest_queue": s.get("_closest_queue") if isinstance(s, dict) else None,
                "closest_published": s.get("_closest_published") if isinstance(s, dict) else None,
            }
            for s in surviving
        ],
        "discarded_ideas": discarded,
        "message": f"{len(ideas)} ideas -> {len(surviving)} surviving, {len(discarded)} discarded (threshold: {threshold})",
        "status": "success",
    })

    return VectorDedupOutput(
        surviving_ideas=surviving,
        discarded_count=len(discarded),
        status="success",
    )


async def find_similar_posts_for_context(
    ctx,
    params: FindSimilarPostsInput,
) -> FindSimilarPostsOutput:
    """
    Find similar posts for RAG context during drafting.

    Used when source_ref_id is NULL to load similar posts via RAG.
    """
    title = params.title
    embedding = params.embedding
    limit = params.limit

    ctx.report_input({
        "title": title,
        "has_embedding": bool(embedding),
        "limit": limit,
    })

    if not embedding:
        ctx.report_output({
            "similar_posts": [],
            "count": 0,
            "status": "no_embedding",
        })
        return FindSimilarPostsOutput(
            similar_posts=[],
            count=0,
            status="no_embedding",
        )

    async with get_db_session() as db:
        # Find similar posts
        result = await db.execute(
            select(Post)
            .where(Post.embedding.isnot(None))
            .order_by(Post.embedding.cosine_distance(embedding))
            .limit(limit)
        )

        items = result.scalars().all()

        similar_posts = [
            SimilarPost(
                id=str(item.id),
                title=item.title,
                content_raw=item.content_raw,
                author=item.author,
                url=item.url,
                likes_count=item.likes_count or 0,
            )
            for item in items
        ]

        logger.info(
            "similar_posts_found",
            title=title[:50],
            count=len(similar_posts),
        )

        # Report output - show actual posts found
        ctx.report_output({
            "similar_posts": [
                {
                    "id": p.id,
                    "title": p.title,
                    "author": p.author,
                    "url": p.url,
                    "likes_count": p.likes_count,
                    "content_preview": p.content_raw[:200] if p.content_raw else None,
                }
                for p in similar_posts
            ],
            "count": len(similar_posts),
            "status": "success",
        })

        return FindSimilarPostsOutput(
            similar_posts=similar_posts,
            count=len(similar_posts),
            status="success",
        )


async def get_style_reference_posts(
    ctx,
    params: GetStyleReferenceInput,
) -> GetStyleReferenceOutput:
    """
    Get high-performing posts as style reference.

    Used to help LLM understand the target writing style.
    """
    publication_handle = params.publication_handle
    limit = params.limit

    ctx.report_input({
        "publication_handle": publication_handle,
        "limit": limit,
    })

    async with get_db_session() as db:
        query = (
            select(Post)
            .order_by(Post.likes_count.desc())
            .limit(limit)
        )

        if publication_handle:
            query = query.where(Post.publication_handle == publication_handle)

        result = await db.execute(query)
        items = result.scalars().all()

        reference_posts = [
            StyleReferencePost(
                title=item.title,
                content_snippet=item.content_raw[:1000] if item.content_raw else "",
                author=item.author,
                likes_count=item.likes_count or 0,
            )
            for item in items
        ]

        logger.info(
            "style_references_found",
            count=len(reference_posts),
            publication_handle=publication_handle,
        )

        # Report output - show actual reference posts
        ctx.report_output({
            "reference_posts": [
                {
                    "title": p.title,
                    "author": p.author,
                    "likes_count": p.likes_count,
                    "content_snippet_preview": p.content_snippet[:200] if p.content_snippet else None,
                }
                for p in reference_posts
            ],
            "count": len(reference_posts),
            "status": "success",
        })

        return GetStyleReferenceOutput(
            reference_posts=reference_posts,
            count=len(reference_posts),
            status="success",
        )


async def check_topic_uniqueness(
    ctx,
    params: CheckTopicUniquenessInput,
) -> CheckTopicUniquenessOutput:
    """
    Quick check if a topic is unique enough to proceed.

    Used as a pre-check before adding to queue.
    """
    topic = params.topic
    embedding = params.embedding
    threshold = params.threshold

    ctx.report_input({
        "topic": topic,
        "has_embedding": bool(embedding),
        "threshold": threshold,
    })

    if not embedding:
        ctx.report_output({
            "is_unique": True,
            "similar_item": None,
            "status": "no_embedding",
        })
        return CheckTopicUniquenessOutput(
            is_unique=True,
            similar_item=None,
            status="no_embedding",
        )

    async with get_db_session() as db:
        # Check production queue
        queue_result = await db.execute(
            select(
                ProductionQueueItem.id,
                ProductionQueueItem.title,
                ProductionQueueItem.embedding.cosine_distance(embedding).label("distance")
            )
            .where(ProductionQueueItem.embedding.isnot(None))
            .order_by("distance")
            .limit(1)
        )
        queue_match = queue_result.fetchone()

        if queue_match and queue_match.distance < threshold:
            similar = SimilarItem(
                type="queue",
                id=str(queue_match.id),
                topic=queue_match.title,
                distance=round(queue_match.distance, 4),
            )
            ctx.report_output({
                "is_unique": False,
                "similar_item": similar.model_dump(),
                "status": "success",
            })
            return CheckTopicUniquenessOutput(
                is_unique=False,
                similar_item=similar,
                status="success",
            )

        # Check target items
        target_result = await db.execute(
            select(
                TargetItem.id,
                TargetItem.title,
                TargetItem.embedding.cosine_distance(embedding).label("distance")
            )
            .where(TargetItem.embedding.isnot(None))
            .order_by("distance")
            .limit(1)
        )
        target_match = target_result.fetchone()

        if target_match and target_match.distance < threshold:
            similar = SimilarItem(
                type="published",
                id=str(target_match.id),
                title=target_match.title,
                distance=round(target_match.distance, 4),
            )
            ctx.report_output({
                "is_unique": False,
                "similar_item": similar.model_dump(),
                "status": "success",
            })
            return CheckTopicUniquenessOutput(
                is_unique=False,
                similar_item=similar,
                status="success",
            )

        ctx.report_output({
            "is_unique": True,
            "similar_item": None,
            "status": "success",
        })
        return CheckTopicUniquenessOutput(
            is_unique=True,
            similar_item=None,
            status="success",
        )
