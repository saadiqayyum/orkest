"""
Node functions for the AI Substack Mirroring Engine.

This package contains all workflow node implementations.
"""

from .db_ops import (
    # Unified posts operations
    upsert_posts,
    identify_new_post_urls,
    get_posts,
    get_posts_needing_content,
    get_posts_needing_embeddings,
    update_post_metrics,
    update_post_embeddings,
    queue_sources_from_posts,
    # Target items
    insert_target_items,
    fetch_my_published_posts,
    # Queue operations
    queue_gap_ideas,
    get_pending_queue_items,
    load_drafting_context,
    update_queue_item_draft,
    set_queue_item_drafting,
    get_queue_item_for_publish,
    finalize_publication,
    # Dedup and topics
    get_existing_topics,
    # Recovery operations
    reset_drafting_status,
    reset_stale_drafting_items,
    # Config loading
    load_target_config,
)

from .scraper import (
    scrape_index,
    scrape_full_content,
    scrape_metrics_batch,
    # New unified posts functions
    fetch_post_listings,
    fetch_post_content_batch,
)

from .embeddings import (
    generate_embedding,
    batch_embed,
    embed_for_dedup,
    batch_embed_posts,
)

from .rag import (
    vector_dedup,
    find_similar_posts_for_context,
    get_style_reference_posts,
    check_topic_uniqueness,
)

from .llm import (
    generate_topic_ideas,
    generate_ideas_from_sources,
    dedup_ideas_llm,
    draft_article,
    generate_image_prompt,
    generate_image,
    refine_draft,
    # Image generation nodes
    analyze_article_for_images,
    generate_all_images,
    stitch_draft,
    embed_image_prompts,
    generate_and_stitch_images,
    # Placeholder injection (legacy)
    inject_placeholders,
    # Unified image injection (v5.3+)
    inject_images,
    generate_images_from_prompts,
)

from .utils import (
    calculate_target_word_count,
)

from .discovery import (
    scrape_leaderboards,
    fetch_publication_posts,
    score_keyword_relevance,
    score_llm_relevance,
    generate_discovery_report,
    generate_report,
    # Publication discovery nodes (v4 - LLM based)
    score_posts_llm,
    rank_publications_llm,
)
