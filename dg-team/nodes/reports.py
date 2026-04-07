"""
Report generation functions for the AI Substack Mirroring Engine.

These functions gather data and format reports about the content pipeline.
"""
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import structlog

from sqlalchemy import select, func

from shared.database import get_db_session
from shared.models import ProductionQueueItem, QueueStatus, Post

logger = structlog.get_logger()


# =============================================================================
# INPUT/OUTPUT SCHEMAS
# =============================================================================

class GatherQueueStatusInput(BaseModel):
    """Input for gather_queue_status function."""
    target_handle: str = Field(..., description="Target publication handle")
    include_items: bool = Field(default=True, description="Include item details")
    items_limit: int = Field(default=10, description="Max items per status")


class QueueCounts(BaseModel):
    """Queue status counts."""
    pending: int = 0
    drafting: int = 0
    review: int = 0
    published: int = 0
    total: int = 0


class QueueItemSummary(BaseModel):
    """Summary of a queue item for the report."""
    id: str
    title: str
    topic_type: str
    priority_score: float
    status: str
    created_at: str


class GatherQueueStatusOutput(BaseModel):
    """Output from gather_queue_status function."""
    status: str = "success"
    queue_data: Dict[str, Any]


class FormatQueueReportInput(BaseModel):
    """Input for format_queue_report function."""
    target_handle: str
    queue_data: Dict[str, Any]


class FormatQueueReportOutput(BaseModel):
    """Output from format_queue_report function."""
    status: str = "success"
    report_title: str
    report_markdown: str
    report_tags: List[str]
    report_json: Dict[str, Any]


class FormatDraftReportInput(BaseModel):
    """Input for format_draft_report function."""
    title: str = Field(..., description="Article title")
    draft_content: str = Field(..., description="Full article content (markdown)")
    image_url: Optional[str] = Field(default=None, description="Hero image URL")
    queue_id: str = Field(..., description="Production queue item ID")
    target_handle: str = Field(..., description="Target publication handle")
    target_word_count: Optional[int] = Field(default=None, description="Target word count")


class FormatDraftReportOutput(BaseModel):
    """Output from format_draft_report function."""
    status: str = "success"
    report_title: str
    report_markdown: str
    report_tags: List[str]
    report_json: Dict[str, Any]


# =============================================================================
# REPORT FUNCTIONS
# =============================================================================

async def gather_queue_status(ctx, params: GatherQueueStatusInput) -> GatherQueueStatusOutput:
    """
    Gather queue status data for a target publication.

    Queries the production_queue table and returns counts and items
    organized by status.
    """
    target_handle = params.target_handle
    include_items = params.include_items
    items_limit = params.items_limit

    logger.info(
        "gathering_queue_status",
        target_handle=target_handle,
        include_items=include_items,
    )

    ctx.report_input({
        "target_handle": target_handle,
        "include_items": include_items,
        "items_limit": items_limit,
    })

    async with get_db_session() as db:
        # Get counts by status
        counts_query = (
            select(
                ProductionQueueItem.status,
                func.count(ProductionQueueItem.id).label("count")
            )
            .where(ProductionQueueItem.target_publication == target_handle)
            .group_by(ProductionQueueItem.status)
        )
        counts_result = await db.execute(counts_query)
        counts_rows = counts_result.fetchall()

        # Build counts dict
        counts = {
            "pending": 0,
            "drafting": 0,
            "review": 0,
            "published": 0,
        }
        for row in counts_rows:
            status_name = row.status.value if hasattr(row.status, 'value') else str(row.status)
            if status_name in counts:
                counts[status_name] = row.count

        counts["total"] = sum(counts.values())

        # Get items by status if requested
        items_by_status = {}
        if include_items:
            for status_val in [QueueStatus.PENDING, QueueStatus.DRAFTING, QueueStatus.REVIEW]:
                status_name = status_val.value

                items_query = (
                    select(ProductionQueueItem)
                    .where(
                        ProductionQueueItem.target_publication == target_handle,
                        ProductionQueueItem.status == status_val,
                    )
                    .order_by(ProductionQueueItem.priority_score.desc())
                    .limit(items_limit)
                )
                items_result = await db.execute(items_query)
                items = items_result.scalars().all()

                items_by_status[status_name] = [
                    {
                        "id": str(item.id),
                        "title": item.title,
                        "topic_type": item.topic_type,
                        "priority_score": item.priority_score or 0.0,
                        "created_at": item.created_at.isoformat() if item.created_at else None,
                    }
                    for item in items
                ]

        # Get posts count for context (total posts in database)
        posts_count_query = select(func.count(Post.id))
        posts_count_result = await db.execute(posts_count_query)
        source_count = posts_count_result.scalar() or 0

    queue_data = {
        "target_handle": target_handle,
        "counts": counts,
        "items_by_status": items_by_status,
        "source_items_total": source_count,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    ctx.report_output({
        "counts": counts,
        "source_items_total": source_count,
    })

    return GatherQueueStatusOutput(
        status="success",
        queue_data=queue_data,
    )


async def format_queue_report(ctx, params: FormatQueueReportInput) -> FormatQueueReportOutput:
    """
    Format queue data into a markdown report.

    Takes the gathered data and produces a formatted markdown document
    suitable for display in the reports browser.
    """
    target_handle = params.target_handle
    queue_data = params.queue_data

    ctx.report_input({
        "target_handle": target_handle,
        "data_keys": list(queue_data.keys()),
    })

    counts = queue_data.get("counts", {})
    items_by_status = queue_data.get("items_by_status", {})
    source_items_total = queue_data.get("source_items_total", 0)
    generated_at = queue_data.get("generated_at", datetime.now(timezone.utc).isoformat())

    # Format the date nicely
    try:
        dt = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        formatted_date = dt.strftime("%B %d, %Y at %I:%M %p UTC")
    except Exception:
        formatted_date = generated_at

    # Build markdown
    lines = []
    lines.append(f"# Queue Status Report")
    lines.append("")
    lines.append(f"**Target:** `{target_handle}`  ")
    lines.append(f"**Generated:** {formatted_date}")
    lines.append("")

    # Summary section
    lines.append("## Summary")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|--------|-------|")
    lines.append(f"| Pending | {counts.get('pending', 0)} |")
    lines.append(f"| Drafting | {counts.get('drafting', 0)} |")
    lines.append(f"| Review | {counts.get('review', 0)} |")
    lines.append(f"| Published | {counts.get('published', 0)} |")
    lines.append(f"| **Total** | **{counts.get('total', 0)}** |")
    lines.append("")
    lines.append(f"*Source items in database: {source_items_total}*")
    lines.append("")

    # Items sections
    status_labels = [
        ("pending", "Pending Items", "Ready to be drafted"),
        ("drafting", "In Progress", "Currently being drafted"),
        ("review", "Under Review", "Drafts ready for review"),
    ]

    for status_key, title, description in status_labels:
        items = items_by_status.get(status_key, [])
        if items:
            lines.append(f"## {title}")
            lines.append(f"*{description}*")
            lines.append("")

            for i, item in enumerate(items, 1):
                item_title = item.get("title", "Unknown title")
                topic_type = item.get("topic_type", "unknown")
                priority = item.get("priority_score", 0)
                lines.append(f"{i}. **{item_title}**")
                lines.append(f"   - Type: {topic_type}")
                lines.append(f"   - Priority: {priority:.2f}")
                lines.append("")

    # No items message
    if counts.get('total', 0) == 0:
        lines.append("## No Items in Queue")
        lines.append("")
        lines.append("Your content queue is empty. Run the **Backlog Manager** workflow to populate it with topics.")
        lines.append("")

    markdown_content = "\n".join(lines)

    # Build title
    report_title = f"Queue Status - {target_handle} - {dt.strftime('%Y-%m-%d')}"

    # Build tags
    report_tags = ["queue-status", target_handle]
    if counts.get('pending', 0) > 0:
        report_tags.append("has-pending")

    # Build JSON data for structured access
    report_json = {
        "target_handle": target_handle,
        "counts": counts,
        "source_items_total": source_items_total,
        "generated_at": generated_at,
    }

    ctx.report_output({
        "title": report_title,
        "tags": report_tags,
        "markdown_length": len(markdown_content),
    })

    return FormatQueueReportOutput(
        status="success",
        report_title=report_title,
        report_markdown=markdown_content,
        report_tags=report_tags,
        report_json=report_json,
    )


async def format_draft_report(ctx, params: FormatDraftReportInput) -> FormatDraftReportOutput:
    """
    Format a draft article into a review report.

    Creates a report that displays the full article content with its hero image,
    making it easy to review formatting, style, and content quality.
    """
    title = params.title
    draft_content = params.draft_content or ""
    image_url = params.image_url
    queue_id = params.queue_id
    target_handle = params.target_handle
    target_word_count = params.target_word_count

    ctx.report_input({
        "title": title,
        "queue_id": queue_id,
        "target_handle": target_handle,
        "has_image": bool(image_url),
        "content_length": len(draft_content),
    })

    generated_at = datetime.now(timezone.utc)
    formatted_date = generated_at.strftime("%B %d, %Y at %I:%M %p UTC")

    # Calculate actual word count
    actual_word_count = len(draft_content.split()) if draft_content else 0

    # Build markdown report
    lines = []

    # Header with metadata
    lines.append(f"# Draft Review: {title}")
    lines.append("")
    lines.append(f"**Generated:** {formatted_date}  ")
    lines.append(f"**Target:** `{target_handle}`  ")
    lines.append(f"**Queue ID:** `{queue_id}`  ")
    lines.append(f"**Word Count:** {actual_word_count:,} words" + (f" (target: {target_word_count:,})" if target_word_count else ""))
    lines.append("")

    # Hero image (if present)
    if image_url:
        lines.append("---")
        lines.append("")
        lines.append("## Hero Image")
        lines.append("")
        lines.append(f"![Hero Image]({image_url})")
        lines.append("")

    # The article content
    lines.append("---")
    lines.append("")
    lines.append("## Article Content")
    lines.append("")
    lines.append(draft_content)
    lines.append("")

    # Review checklist
    lines.append("---")
    lines.append("")
    lines.append("## Review Checklist")
    lines.append("")
    lines.append("- [ ] Title is engaging and accurate")
    lines.append("- [ ] Opening hook captures attention")
    lines.append("- [ ] Content flows logically")
    lines.append("- [ ] Tone matches brand voice")
    lines.append("- [ ] No factual errors or hallucinations")
    lines.append("- [ ] Hero image fits the content")
    lines.append("- [ ] Ready for publication")
    lines.append("")

    markdown_content = "\n".join(lines)

    # Build title (short version for list view)
    report_title = f"Draft: {title[:50]}{'...' if len(title) > 50 else ''}"

    # Build tags
    report_tags = ["draft-review", target_handle, "needs-review"]
    if image_url:
        report_tags.append("has-image")

    # Build JSON data for structured access
    report_json = {
        "title": title,
        "queue_id": queue_id,
        "target_handle": target_handle,
        "actual_word_count": actual_word_count,
        "target_word_count": target_word_count,
        "has_image": bool(image_url),
        "image_url": image_url,
        "generated_at": generated_at.isoformat(),
    }

    ctx.report_output({
        "title": report_title,
        "tags": report_tags,
        "word_count": actual_word_count,
        "has_image": bool(image_url),
    })

    return FormatDraftReportOutput(
        status="success",
        report_title=report_title,
        report_markdown=markdown_content,
        report_tags=report_tags,
        report_json=report_json,
    )
