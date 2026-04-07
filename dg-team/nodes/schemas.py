"""
Pydantic schemas for workflow node inputs and outputs.

These provide type safety and validation for all node functions,
similar to LangGraph's typed state approach.
"""
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from uuid import UUID


# =============================================================================
# LLM MODEL REGISTRY
# =============================================================================
# Maps user-friendly model names to provider + API model ID
# Format: "Display Name" -> (provider, model_id)

LLM_MODEL_REGISTRY: Dict[str, tuple] = {
    # Google Gemini
    "Gemini 3 Pro": ("gemini", "gemini-3-pro-preview"),
    "Gemini 3 Flash": ("gemini", "gemini-3-flash-preview"),
    "Gemini 2.5 Pro": ("gemini", "gemini-2.5-pro"),
    "Gemini 2.5 Flash": ("gemini", "gemini-2.5-flash"),
    "Gemini 2.5 Flash Lite": ("gemini", "gemini-2.5-flash-lite"),  # Cost-efficient, no thinking by default
    "Gemini 2.0 Flash": ("gemini", "gemini-2.0-flash-exp"),  # Deprecated March 2026
    # Gemini 1.5 models retired April 2025
    # "Gemini 1.5 Pro": ("gemini", "gemini-1.5-pro"),
    # "Gemini 1.5 Flash": ("gemini", "gemini-1.5-flash"),

    # Anthropic Claude
    "Claude Opus 4.5": ("anthropic", "claude-opus-4-5-20251101"),
    "Claude Sonnet 4.5": ("anthropic", "claude-sonnet-4-5-20251101"),
    "Claude Opus 4": ("anthropic", "claude-opus-4-20250514"),
    "Claude Sonnet 4": ("anthropic", "claude-sonnet-4-20250514"),
    "Claude Sonnet 3.7": ("anthropic", "claude-3-7-sonnet-20250219"),
    "Claude 3 Opus": ("anthropic", "claude-3-opus-20240229"),
    "Claude 3.5 Haiku": ("anthropic", "claude-3-5-haiku-20241022"),

    # OpenAI
    "GPT-5": ("openai", "gpt-5"),
    "GPT-5 Mini": ("openai", "gpt-5-mini"),
    "GPT-5 Nano": ("openai", "gpt-5-nano"),
    "GPT-4.1": ("openai", "gpt-4.1"),
    "GPT-4o": ("openai", "gpt-4o"),
    "GPT-4o Mini": ("openai", "gpt-4o-mini"),
    "o3": ("openai", "o3"),
    "o3 Mini": ("openai", "o3-mini"),
    "o1": ("openai", "o1"),

    # Local (Ollama)
    "Llama 3.1 (Local)": ("ollama", "llama3.1"),
    "Llama 3.2 (Local)": ("ollama", "llama3.2:3b"),
    "Mistral (Local)": ("ollama", "mistral"),
    "Qwen 2.5 (Local)": ("ollama", "qwen2.5"),
    "Qwen 2.5 Coder 32B (Local)": ("ollama", "qwen2.5-coder:32b"),
}

# List of model names for enum validation
LLM_MODEL_CHOICES = list(LLM_MODEL_REGISTRY.keys())


def resolve_model(model_name: Optional[str]) -> tuple:
    """
    Resolve a model name to (provider, model_id).

    Args:
        model_name: User-friendly model name (e.g., "Gemini 3 Pro")

    Returns:
        Tuple of (provider, model_id) or (None, None) if not found
    """
    if not model_name:
        return (None, None)
    return LLM_MODEL_REGISTRY.get(model_name, (None, None))


# =============================================================================
# LLM CONFIGURATION (Shared across all LLM nodes)
# =============================================================================

class LLMConfig(BaseModel):
    """
    LLM configuration that can be passed to any LLM node.

    Uses a single model selector mapped via LLM_MODEL_REGISTRY.

    Resolution order (first non-None wins):
    1. Node params (this object)
    2. Workflow trigger input
    3. Environment variables (.env)
    4. Defaults (Llama 3.1 Local)

    Example workflow usage:
        params:
          llm_config:
            model: "Gemini 3 Pro"
            temperature: 0
    """
    model: Optional[str] = Field(
        default=None,
        description="Model to use (e.g., 'Gemini 3 Pro', 'Claude Opus 4', 'GPT-4o')"
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0,
        le=2,
        description="Temperature for LLM sampling (0=deterministic, 1=default, 2=max creativity)"
    )


class ImageConfig(BaseModel):
    """
    Image generation provider configuration.

    Example workflow usage:
        params:
          image_config:
            image_provider: openai
    """
    image_provider: Optional[str] = Field(
        default=None,
        description="Image provider: 'placeholder', 'openai', 'stable-diffusion', 'gemini'"
    )


# =============================================================================
# SCRAPER SCHEMAS
# =============================================================================

class ScrapeIndexInput(BaseModel):
    """Input for scrape_index node."""
    source_handles: List[str] = Field(
        default_factory=list,
        description="List of Substack handles to scrape (e.g., ['stratechery', 'designgurus'])"
    )
    target_handle: str = Field(
        default="",
        description="Target publication handle (e.g., 'postsyntax')"
    )

    @field_validator('source_handles', mode='before')
    @classmethod
    def coerce_to_list(cls, v):
        """Accept both string and list for source_handles."""
        if isinstance(v, str):
            return [v] if v else []
        return v


class PostMetadata(BaseModel):
    """Metadata for a scraped post from Substack API."""
    url: str
    title: str
    author: Optional[str] = None
    publication_handle: str
    likes_count: int = 0
    comments_count: int = 0
    shares_count: Optional[int] = None
    published_at: Optional[str] = None
    # Audience from Substack API: 'everyone' (free), 'only_paid', 'founding', or None
    audience: Optional[str] = None
    # Additional fields from Substack API
    post_id: Optional[str] = None  # Substack's numeric post ID
    subtitle: Optional[str] = None  # Post subtitle (merged with description - stores whichever is longer)
    post_type: Optional[str] = None  # 'newsletter', 'podcast', 'thread'
    word_count: Optional[int] = None  # Approximate word count


class ScrapeIndexOutput(BaseModel):
    """Output from scrape_index node (fetch_post_listings)."""
    source_posts: List[PostMetadata] = Field(default_factory=list)
    target_posts: List[PostMetadata] = Field(default_factory=list)
    status: str = "success"


class ScrapeFullContentInput(BaseModel):
    """Input for scrape_full_content node."""
    urls: List[str] = Field(
        default_factory=list,
        description="List of post URLs to scrape full content from"
    )
    # Full post objects with metadata (preferred - preserves publication_handle)
    posts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Post objects with url, title, publication_handle, etc."
    )
    publication_handle: Optional[str] = Field(
        default=None,
        description="Optional handle if all URLs are from same publication"
    )
    # Authentication for paid content
    substack_cookies: Optional[Dict[str, str]] = Field(
        default=None,
        description="Substack session cookies for accessing paid subscriber content (substack.sid, substack.lli)"
    )
    # Whitelist of publications where we MUST get full content
    paid_subscription_handles: List[str] = Field(
        default_factory=list,
        description="Publications where we have paid access - fail if can't get full content"
    )


class ScrapedItem(BaseModel):
    """A fully scraped item with content."""
    url: str
    title: str
    content_raw: Optional[str] = None
    author: Optional[str] = None
    publication_handle: Optional[str] = None
    published_at: Optional[str] = None
    # Engagement metrics (optional - populated from index scrape, not full content scrape)
    likes_count: int = 0
    comments_count: int = 0
    # Paywall tracking - to know if we have full content or partial
    is_paywalled: bool = False  # True if post is behind paywall
    has_full_content: bool = True  # False if we only got partial content (paywall blocked)


class ScrapeFullContentOutput(BaseModel):
    """Output from scrape_full_content node."""
    items: List[ScrapedItem] = Field(default_factory=list)
    failed_urls: List[str] = Field(default_factory=list)
    status: str = "success"


class ScrapeMetricsBatchInput(BaseModel):
    """Input for scrape_metrics_batch node."""
    items: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of items with 'url' and 'publication_handle'"
    )


class MetricsUpdate(BaseModel):
    """Updated metrics for a post."""
    url: str
    likes_count: int = 0
    comments_count: int = 0
    shares_count: Optional[int] = None


class ScrapeMetricsBatchOutput(BaseModel):
    """Output from scrape_metrics_batch node."""
    updates: List[MetricsUpdate] = Field(default_factory=list)
    count: int = 0
    status: str = "success"


# =============================================================================
# DB_OPS SCHEMAS - Harvester
# =============================================================================

class FilterNewUrlsInput(BaseModel):
    """Input for identify_new_posts node."""
    target_handle: str = Field(default="", description="Target publication handle for scoping")
    # Post objects from scrape_index (contains URL, title, publication_handle, metrics, etc.)
    source_posts: List[Dict[str, Any]] = Field(default_factory=list)
    target_posts: List[Dict[str, Any]] = Field(default_factory=list)
    # Whitelist of publications where we have paid subscriptions
    paid_subscription_handles: List[str] = Field(default_factory=list)


class FilterNewUrlsOutput(BaseModel):
    """Output from identify_new_posts node."""
    # New posts not yet in DB (need content fetch and storage)
    new_source_posts: List[Dict[str, Any]] = Field(default_factory=list)
    new_target_posts: List[Dict[str, Any]] = Field(default_factory=list)
    status: str = "success"


class InsertTargetItemsInput(BaseModel):
    """Input for insert_target_items node."""
    target_handle: str = Field(default="", description="Target publication handle for scoping")
    items: List[ScrapedItem] = Field(default_factory=list)


class InsertTargetItemsOutput(BaseModel):
    """Output from insert_target_items node."""
    inserted_count: int = 0
    item_ids: List[str] = Field(default_factory=list)
    status: str = "success"


# =============================================================================
# DB_OPS SCHEMAS - Backlog Manager
# =============================================================================

class FindUniqueCandidatesInput(BaseModel):
    """Input for find_unique_candidates node - combined hard + soft dedup."""
    target_handle: str = Field(default="", description="Target publication handle for scoping")
    min_likes: int = Field(default=5, description="Minimum likes threshold")
    max_age_days: int = Field(default=365, description="Maximum age in days")
    source_picks: int = Field(default=3, description="Number of unique candidates to find")
    dedup_threshold: float = Field(default=0.15, description="Cosine distance threshold for soft dedup")
    batch_size: int = Field(default=50, description="Batch size for scanning")


class UniqueCandidate(BaseModel):
    """A unique candidate that passed both hard and soft dedup."""
    id: str
    url: str
    title: str
    author: Optional[str] = None
    publication_handle: str
    likes_count: int = 0
    comments_count: int = 0
    published_at: Optional[str] = None
    priority_score: float = 0.0


class FindUniqueCandidatesOutput(BaseModel):
    """Output from find_unique_candidates node."""
    candidates: List[UniqueCandidate] = Field(default_factory=list)
    count: int = 0
    scanned_count: int = 0
    hard_excluded_count: int = 0
    soft_excluded_count: int = 0
    status: str = "success"


class QueueSourceItemsInput(BaseModel):
    """Input for queue_source_items node (legacy - expects pre-filtered items)."""
    target_handle: str = Field(default="", description="Target publication handle")
    items: List[UniqueCandidate] = Field(default_factory=list, description="Candidates to queue")


class QueueSourcesInput(BaseModel):
    """Input for queue_sources node - auto-picks and queues source articles."""
    target_handle: str = Field(default="", description="Target publication handle")
    count: int = Field(default=10, description="Number of articles to queue")
    min_likes: int = Field(default=10, description="Minimum likes threshold")
    max_age_days: int = Field(default=90, description="Only consider articles published within this many days")
    recent_days: int = Field(default=7, description="Articles within this many days are considered 'recent'")


class QueuedSourceItem(BaseModel):
    """A source item that was queued."""
    queue_id: str
    source_id: str
    title: str


class QueueSourceItemsOutput(BaseModel):
    """Output from queue_source_items node."""
    queued_count: int = 0
    queue_ids: List[str] = Field(default_factory=list)
    queued_items: List[QueuedSourceItem] = Field(default_factory=list)
    status: str = "success"


class QueueSourcesOutput(BaseModel):
    """Output from queue_sources node."""
    queued_count: int = 0
    queued_from_recent: int = 0
    queued_from_top: int = 0
    skipped_already_queued: int = 0
    skipped_already_published: int = 0
    eligible_count: int = 0
    queued_items: List[QueuedSourceItem] = Field(default_factory=list)
    status: str = "success"


# =============================================================================
# DB_OPS SCHEMAS - Idea Generation
# =============================================================================

class ResolveSourcePublicationsInput(BaseModel):
    """Input for resolve_source_publications node."""
    publications: Optional[List[str]] = Field(default=None, description="Direct list of source publication handles")
    discovery_kv_key: Optional[str] = Field(default=None, description="KV key to fetch publications from (from discovery workflow)")


class ResolveSourcePublicationsOutput(BaseModel):
    """Output from resolve_source_publications node."""
    publications: List[str] = Field(default_factory=list, description="Resolved list of publication handles")
    source: str = Field(default="", description="Where publications came from: 'input' or 'kv'")
    count: int = 0
    status: str = "success"


class FetchTopSourcePostsInput(BaseModel):
    """Input for fetch_top_source_posts node."""
    publications: List[str] = Field(default_factory=list, description="List of source publication handles")
    posts_per_source: int = Field(default=5, description="Number of top posts per source publication")
    max_age_days: int = Field(default=30, description="Only posts from last N days")
    target_handle: str = Field(default="", description="Target publication handle (for filtering out already-worked posts)")


class SourcePost(BaseModel):
    """A source post with engagement data and context for LLM idea generation."""
    id: str  # Needed for dedup against queue/published
    title: str
    publication_handle: str
    likes_count: int = 0
    # Additional context for better LLM idea generation
    subtitle: Optional[str] = None  # Merged subtitle/description - stores whichever is longer
    tags: List[str] = Field(default_factory=list)
    section_name: Optional[str] = None
    word_count: Optional[int] = None
    comments_count: int = 0


class FetchTopSourcePostsOutput(BaseModel):
    """Output from fetch_top_source_posts node."""
    # Use Dict for JSON serialization in workflow context (Pydantic objects fail)
    source_posts: List[Dict[str, Any]] = Field(default_factory=list)
    total_count: int = 0
    sources_count: int = 0
    status: str = "success"


class FetchMyPublishedPostsInput(BaseModel):
    """Input for fetch_my_published_posts node."""
    target_handle: str = Field(default="", description="Target publication handle")
    limit: int = Field(default=10, description="Number of posts to fetch")
    content_preview_length: int = Field(default=500, description="Characters of content to include")


class MyPublishedPost(BaseModel):
    """One of my published posts with content preview. Minimal fields for token efficiency."""
    id: str = ""
    url: str = ""
    title: str
    content_preview: str = ""
    likes_count: int = 0
    published_at: Optional[str] = None


class FetchMyPublishedPostsOutput(BaseModel):
    """Output from fetch_my_published_posts node."""
    # Use Dict for JSON serialization in workflow context (Pydantic objects fail)
    my_posts: List[Dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    status: str = "success"


class FilterWorkedSourcePostsInput(BaseModel):
    """Input for filter_worked_source_posts node."""
    target_handle: str = Field(default="", description="Target publication handle")
    # Accept dicts (JSON-serialized from workflow context)
    posts: List[Dict[str, Any]] = Field(default_factory=list, description="Source posts to filter")


class FilterWorkedSourcePostsOutput(BaseModel):
    """Output from filter_worked_source_posts node."""
    # Use Dict for JSON serialization in workflow context (Pydantic objects fail)
    source_posts: List[Dict[str, Any]] = Field(default_factory=list)
    filtered_count: int = 0
    removed_count: int = 0
    status: str = "success"


class GenerateIdeasFromSourcesInput(BaseModel):
    """Input for generate_ideas_from_sources LLM node."""
    # Accept dicts (JSON-serialized from workflow context)
    source_posts: List[Dict[str, Any]] = Field(default_factory=list, description="Top source posts")
    my_posts: List[Dict[str, Any]] = Field(default_factory=list, description="My published posts for voice")
    idea_count: int = Field(default=5, description="Number of ideas to generate")
    llm_config: Optional[LLMConfig] = None


class GeneratedIdea(BaseModel):
    """A generated topic idea."""
    title: str
    summary: str
    post_type: str = "free"  # "free" or "paid"
    inspiration_source: Optional[str] = None  # Human-readable description of inspiration
    source_post_ids: List[str] = Field(default_factory=list)  # UUIDs of source posts that inspired this idea


class GenerateIdeasFromSourcesOutput(BaseModel):
    """Output from generate_ideas_from_sources node."""
    ideas: List[GeneratedIdea] = Field(default_factory=list)
    count: int = 0
    status: str = "success"


# =============================================================================
# LLM SCHEMAS - Idea Deduplication
# =============================================================================

class DedupIdeasLLMInput(BaseModel):
    """Input for dedup_ideas_llm node."""
    target_handle: str = Field(default="", description="Target publication handle")
    ideas: List[Dict[str, Any]] = Field(default_factory=list, description="New ideas to check")
    llm_config: Optional[LLMConfig] = None


class DedupResult(BaseModel):
    """Result for a single idea dedup check."""
    title: str
    similarity_score: int = Field(default=0, ge=0, le=100, description="0-100 similarity score")
    is_duplicate: bool
    similar_to: Optional[str] = None
    reason: str = ""  # Brief explanation of decision


class DedupIdeasLLMOutput(BaseModel):
    """Output from dedup_ideas_llm node."""
    surviving_ideas: List[Dict[str, Any]] = Field(default_factory=list)
    discarded_ideas: List[DedupResult] = Field(default_factory=list)
    surviving_count: int = 0
    discarded_count: int = 0
    status: str = "success"


# =============================================================================
# DB_OPS SCHEMAS - Metrics Update
# =============================================================================

class GetStaleMetricsItemsInput(BaseModel):
    """Input for get_source_items_with_stale_metrics node."""
    stale_days: int = Field(default=7, description="Refresh metrics if not updated in this many days")


class StaleMetricsItem(BaseModel):
    """An item with stale metrics that needs refreshing."""
    url: str
    publication_handle: str
    metrics_updated_at: Optional[str] = None  # ISO timestamp or None if never updated


class GetStaleMetricsItemsOutput(BaseModel):
    """Output from get_source_items_with_stale_metrics node."""
    items: List[StaleMetricsItem] = Field(default_factory=list)
    count: int = 0
    status: str = "success"


class UpdateSourceMetricsInput(BaseModel):
    """Input for update_source_metrics node."""
    updates: List[MetricsUpdate] = Field(default_factory=list)


class UpdateSourceMetricsOutput(BaseModel):
    """Output from update_source_metrics node."""
    updated_count: int = 0
    status: str = "success"


# =============================================================================
# EMBEDDING SCHEMAS
# =============================================================================

class GenerateEmbeddingInput(BaseModel):
    """Input for generate_embedding node."""
    text: str = Field(default="", description="Text to embed")


class GenerateEmbeddingOutput(BaseModel):
    """Output from generate_embedding node."""
    embedding: List[float] = Field(default_factory=list)
    dimensions: int = 0
    status: str = "success"


class EmbeddingItem(BaseModel):
    """An item for batch embedding."""
    id: Optional[str] = None
    title: Optional[str] = None
    content_raw: Optional[str] = None
    # Allow extra fields for flexibility
    model_config = {"extra": "allow"}


class BatchEmbedInput(BaseModel):
    """Input for batch_embed node."""
    # Use Dict[str, Any] to accept any item structure - the text_field param specifies which field to use
    items: List[Dict[str, Any]] = Field(default_factory=list)
    text_field: str = Field(default="title", description="Field name containing text")


class EmbeddedItem(BaseModel):
    """Result of embedding an item."""
    id: Optional[str] = None
    embedding: List[float] = Field(default_factory=list)
    text: str = ""
    # Original item data (for pass-through)
    original: Optional[Dict[str, Any]] = None
    # Allow extra fields for flexibility
    model_config = {"extra": "allow"}


class BatchEmbedOutput(BaseModel):
    """Output from batch_embed node."""
    # Items with embeddings + original data preserved
    items: List[Dict[str, Any]] = Field(default_factory=list)
    success_count: int = 0
    failed_count: int = 0
    status: str = "success"


class BatchUpdateEmbeddingsInput(BaseModel):
    """Input for batch_update_embeddings node."""
    item_type: str = Field(default="source", description="Type: 'source', 'target', or 'queue'")
    items: List[Dict[str, Any]] = Field(default_factory=list, description="Items with 'id' and 'embedding'")


class BatchUpdateEmbeddingsOutput(BaseModel):
    """Output from batch_update_embeddings node."""
    updated_count: int = 0
    status: str = "success"


# =============================================================================
# RAG SCHEMAS
# =============================================================================

class IdeaForDedup(BaseModel):
    """An idea with embedding for deduplication."""
    title: str
    embedding: List[float] = Field(default_factory=list)
    rationale: Optional[str] = None


class VectorDedupInput(BaseModel):
    """Input for vector_dedup node."""
    # Accept any dict with title and embedding fields
    ideas: List[Dict[str, Any]] = Field(default_factory=list)
    threshold: float = Field(default=0.15, description="Cosine distance threshold")
    max_results: Optional[int] = Field(default=None, description="Max surviving ideas to return (caps output)")


class VectorDedupOutput(BaseModel):
    """Output from vector_dedup node."""
    surviving_ideas: List[Dict[str, Any]] = Field(default_factory=list)
    discarded_count: int = 0
    status: str = "success"


class FindSimilarPostsInput(BaseModel):
    """Input for find_similar_posts_for_context node."""
    title: str = ""
    embedding: List[float] = Field(default_factory=list)
    limit: int = Field(default=3, description="Number of similar posts to retrieve")


class SimilarPost(BaseModel):
    """A similar post for context."""
    id: str
    title: str
    content_raw: Optional[str] = None
    author: Optional[str] = None
    url: str
    likes_count: int = 0


class FindSimilarPostsOutput(BaseModel):
    """Output from find_similar_posts_for_context node."""
    similar_posts: List[SimilarPost] = Field(default_factory=list)
    count: int = 0
    status: str = "success"


class GetStyleReferenceInput(BaseModel):
    """Input for get_style_reference_posts node."""
    publication_handle: Optional[str] = None
    limit: int = Field(default=5, description="Number of posts to retrieve")


class StyleReferencePost(BaseModel):
    """A style reference post."""
    title: str
    content_snippet: str = ""
    author: Optional[str] = None
    likes_count: int = 0


class GetStyleReferenceOutput(BaseModel):
    """Output from get_style_reference_posts node."""
    reference_posts: List[StyleReferencePost] = Field(default_factory=list)
    count: int = 0
    status: str = "success"


class CheckTopicUniquenessInput(BaseModel):
    """Input for check_topic_uniqueness node."""
    topic: str = ""
    embedding: List[float] = Field(default_factory=list)
    threshold: float = Field(default=0.15, description="Similarity threshold")


class SimilarItem(BaseModel):
    """A similar item found during uniqueness check."""
    type: str  # "queue" or "published"
    id: str
    topic: Optional[str] = None
    title: Optional[str] = None
    distance: float = 0.0


class CheckTopicUniquenessOutput(BaseModel):
    """Output from check_topic_uniqueness node."""
    is_unique: bool = True
    similar_item: Optional[SimilarItem] = None
    status: str = "success"


# =============================================================================
# LLM SCHEMAS
# =============================================================================

class SourcePattern(BaseModel):
    """A source pattern for context."""
    title: str
    likes_count: int = 0


class GenerateTopicIdeasInput(BaseModel):
    """Input for generate_topic_ideas node."""
    count: int = Field(default=3, description="Number of ideas to generate (max returned)")
    source_patterns: List[SourcePattern] = Field(default_factory=list)
    existing_topics: List[str] = Field(default_factory=list, description="Existing topics to avoid")
    # LLM config override (optional - falls back to workflow config, then .env)
    llm_config: Optional[LLMConfig] = Field(default=None, description="Override LLM provider/model")


class TopicIdea(BaseModel):
    """A generated topic idea."""
    title: str
    rationale: str = ""


class GenerateTopicIdeasOutput(BaseModel):
    """Output from generate_topic_ideas node."""
    topic_ideas: List[TopicIdea] = Field(default_factory=list)
    count: int = 0
    status: str = "success"


class ReferencePost(BaseModel):
    """A reference post for style."""
    title: str = ""
    content_snippet: str = ""


class DraftConstraints(BaseModel):
    """Constraints for article drafting."""
    word_count: int = 1500
    avoid: List[str] = Field(default_factory=list)


class DraftArticleInput(BaseModel):
    """Input for draft_article node."""
    title: str = ""
    source_content: Optional[str] = None
    reference_posts: List[ReferencePost] = Field(default_factory=list)
    article_prompt: str = Field(default="", description="Writing style instructions for the LLM")
    constraints: Optional[DraftConstraints] = None
    inline_images: int = Field(default=0, description="Number of inline images to suggest (0-3)")
    image_mode: str = Field(default="generate", description="'generate' for actual images, 'prompts_only' for placeholders")
    # Rich context from idea generation (for original ideas, not source-derived)
    idea_summary: Optional[str] = Field(default=None, description="Detailed summary of the idea")
    idea_inspiration: Optional[str] = Field(default=None, description="Which source post inspired this idea")
    post_type: Optional[str] = Field(default=None, description="'paid' or 'free' - affects depth/exclusivity")
    # LLM config override (optional - falls back to workflow config, then .env)
    llm_config: Optional[LLMConfig] = Field(default=None, description="Override LLM provider/model")


class DraftArticleOutput(BaseModel):
    """Output from draft_article node."""
    draft_content: str = ""
    word_count: int = 0
    status: str = "success"
    article_prompt: Optional[str] = None


class GenerateImagePromptInput(BaseModel):
    """Input for generate_image_prompt node."""
    article_content: str = ""
    title: str = ""
    image_prompt: str = Field(default="", description="Style instructions for image generation")
    # LLM config override (optional - falls back to workflow config, then .env)
    llm_config: Optional[LLMConfig] = Field(default=None, description="Override LLM provider/model")


class GenerateImagePromptOutput(BaseModel):
    """Output from generate_image_prompt node."""
    final_image_prompt: str = ""
    status: str = "success"


class EmbedImagePromptsInput(BaseModel):
    """Input for embed_image_prompts node (prompts_only mode)."""
    draft_content: str = ""
    hero_image_prompt: str = ""
    inline_image_prompts: List[Dict[str, str]] = Field(default_factory=list, description="List of {after_section, prompt} dicts")
    hero_image_style: str = Field(default="", description="Style prefix for hero image (from settings)")
    inline_image_style: str = Field(default="", description="Style prefix for inline images (from settings)")
    image_model: str = Field(default="gemini-2.0-flash-exp", description="Recommended model for generating images")


class EmbedImagePromptsOutput(BaseModel):
    """Output from embed_image_prompts node."""
    draft_content: str = ""  # Modified content with embedded prompts
    status: str = "success"


# =============================================================================
# PLACEHOLDER INJECTION SCHEMAS (Legacy - kept for backwards compatibility)
# =============================================================================

class ImagePromptOutput(BaseModel):
    """Final compiled image prompt for a single image."""
    placeholder: str = Field(description="The placeholder tag, e.g., '<image:1>'")
    placement: str = Field(description="Where the image goes: 'hero' or 'after: Section Name'")
    description_prompt: str = Field(description="The LLM-generated description of what to show")
    final_prompt: str = Field(description="Complete prompt: base_style + description")
    alt_text: str = Field(default="", description="Accessibility alt text")


class InjectPlaceholdersInput(BaseModel):
    """Input for inject_placeholders node (legacy)."""
    draft_content: str = Field(description="The raw article content")
    image_analysis: Optional["ImageAnalysisResult"] = Field(default=None, description="Image analysis with placement info")
    post_type: Optional[str] = Field(default=None, description="'paid' or 'free' - determines paywall insertion")
    hero_image_style: str = Field(default="", description="Base style prompt for hero image")
    inline_image_style: str = Field(default="", description="Base style prompt for inline images")


class InjectPlaceholdersOutput(BaseModel):
    """Output from inject_placeholders node (legacy)."""
    draft_content: str = Field(default="", description="Article with <image:N> and <paywall> placeholders")
    image_prompts: List[ImagePromptOutput] = Field(default_factory=list, description="Compiled image prompts")
    has_paywall: bool = Field(default=False, description="Whether paywall was inserted")
    status: str = "success"


# =============================================================================
# NEW IMAGE INJECTION SCHEMAS (v5.3+) - LLM inserts placeholders directly
# =============================================================================

class ImagePromptItem(BaseModel):
    """Image prompt keyed by placeholder name."""
    prompt: str = Field(description="The image generation prompt")
    alt_text: str = Field(default="", description="Accessibility alt text")


class LLMInjectImagesResponse(BaseModel):
    """Response schema for LLM that injects images directly into article."""
    modified_article: str = Field(description="The article with <image:1>, <image:2>, etc. and optionally <paywall> inserted")
    image_prompts: Dict[str, ImagePromptItem] = Field(
        description="Image prompts keyed by placeholder name, e.g., {'image:1': {...}, 'image:2': {...}}"
    )


class InjectImagesInput(BaseModel):
    """Input for inject_images node (new unified approach)."""
    draft_content: str = Field(description="The raw article content")
    post_type: Optional[str] = Field(default=None, description="'paid' or 'free' - determines paywall insertion")
    min_images: int = Field(default=1, description="Minimum number of images (including hero)")
    max_images: int = Field(default=3, description="Maximum number of images")
    # LLM config override
    llm_config: Optional[LLMConfig] = Field(default=None, description="Override LLM provider/model")


class CompiledImagePrompt(BaseModel):
    """Final compiled image prompt ready for image generation."""
    placeholder: str = Field(description="e.g., 'image:1'")
    description_prompt: str = Field(description="LLM-generated description")
    final_prompt: str = Field(description="base_style + description")
    alt_text: str = Field(default="")


class InjectImagesOutput(BaseModel):
    """Output from inject_images node."""
    draft_content: str = Field(default="", description="Article with <image:N> and <paywall> placeholders")
    image_prompts: Dict[str, CompiledImagePrompt] = Field(
        default_factory=dict,
        description="Image prompts keyed by placeholder, e.g., {'image:1': {...}}"
    )
    has_paywall: bool = Field(default=False)
    status: str = "success"


class GenerateImageInput(BaseModel):
    """Input for generate_image node."""
    image_prompt: str = ""
    # Image config override (optional - falls back to workflow config, then .env)
    image_config: Optional[ImageConfig] = Field(default=None, description="Override image provider")


class GenerateImageOutput(BaseModel):
    """Output from generate_image node."""
    image_url: Optional[str] = None
    status: str = "success"
    reason: Optional[str] = None


class GeneratedImageResult(BaseModel):
    """Single generated image result."""
    placeholder: str = Field(description="e.g., 'image:1'")
    image_url: str = Field(description="URL of generated image")
    prompt_used: str = Field(default="", description="The prompt used to generate")


class GenerateImagesFromPromptsInput(BaseModel):
    """Input for generate_images_from_prompts node."""
    image_prompts: Dict[str, CompiledImagePrompt] = Field(
        default_factory=dict,
        description="Image prompts keyed by placeholder from inject_images"
    )
    image_model: Optional[str] = Field(default=None, description="Image model override")
    image_config: Optional[ImageConfig] = Field(default=None, description="Image config override")
    generate_images: bool = Field(default=False, description="Whether to actually generate images")


class GenerateImagesFromPromptsOutput(BaseModel):
    """Output from generate_images_from_prompts node."""
    generated_images: Dict[str, GeneratedImageResult] = Field(
        default_factory=dict,
        description="Generated images keyed by placeholder"
    )
    total_generated: int = 0
    total_failed: int = 0
    status: str = "success"


class RefineDraftInput(BaseModel):
    """Input for refine_draft node."""
    draft_content: str = ""
    feedback: str = ""
    # LLM config override (optional - falls back to workflow config, then .env)
    llm_config: Optional[LLMConfig] = Field(default=None, description="Override LLM provider/model")


class RefineDraftOutput(BaseModel):
    """Output from refine_draft node."""
    refined_content: str = ""
    word_count: int = 0
    status: str = "success"


# =============================================================================
# PUBLISHER SCHEMAS
# =============================================================================

class PrepareForPublishInput(BaseModel):
    """Input for prepare_for_manual_publish node."""
    queue_id: str = ""
    title: str = ""
    content: str = ""
    image_url: Optional[str] = None


class ReviewPackage(BaseModel):
    """A review package for manual publishing."""
    queue_id: str
    title: str
    content: str
    image_url: Optional[str] = None
    publish_url: str = ""
    instructions: List[str] = Field(default_factory=list)


class PrepareForPublishOutput(BaseModel):
    """Output from prepare_for_manual_publish node."""
    review_package: ReviewPackage
    status: str = "success"


class RecordPublicationInput(BaseModel):
    """Input for record_manual_publication node."""
    queue_id: str = ""
    published_url: str = ""
    published_title: str = ""


class RecordPublicationOutput(BaseModel):
    """Output from record_manual_publication node."""
    target_item_id: Optional[str] = None
    status: str = "success"


# =============================================================================
# ADDITIONAL DB_OPS SCHEMAS
# =============================================================================

class QueueGapIdeasInput(BaseModel):
    """Input for queue_gap_ideas node."""
    target_handle: str = Field(default="", description="Target publication handle")
    # Accept dicts - ideas come from vector_dedup with varied fields (title, summary, post_type, embedding, etc.)
    ideas: List[Dict[str, Any]] = Field(default_factory=list, description="Ideas to queue")


class QueuedIdea(BaseModel):
    """A queued idea."""
    queue_id: str
    title: str


class QueueGapIdeasOutput(BaseModel):
    """Output from queue_gap_ideas node."""
    queued_count: int = 0
    queue_ids: List[str] = Field(default_factory=list)
    queued_items: List[QueuedIdea] = Field(default_factory=list)
    status: str = "success"


class GetExistingTopicsInput(BaseModel):
    """Input for get_existing_topics node."""
    target_handle: str = Field(..., description="Target publication handle")
    limit: int = Field(default=50, description="Max topics to return")


class GetExistingTopicsOutput(BaseModel):
    """Output from get_existing_topics node."""
    existing_topics: List[str] = Field(default_factory=list, description="List of titles")
    count: int = 0
    status: str = "success"


class GetPendingQueueItemsInput(BaseModel):
    """Input for get_pending_queue_items node."""
    target_handle: str = Field(default="", description="Target publication handle")
    limit: int = Field(default=1, description="Max items to return")
    queue_id: Optional[str] = Field(default=None, description="Specific queue item ID (overrides pending lookup)")


class PendingQueueItem(BaseModel):
    """A pending queue item."""
    id: str
    source_ref_id: Optional[str] = None
    title: str
    topic_type: str
    priority_score: float = 0.0
    # Rich context from idea generation
    idea_summary: Optional[str] = None  # Detailed summary of the idea
    idea_inspiration: Optional[str] = None  # What source post inspired this
    post_type: Optional[str] = None  # "paid" or "free"


class GetPendingQueueItemsOutput(BaseModel):
    """Output from get_pending_queue_items node."""
    items: List[PendingQueueItem] = Field(default_factory=list)
    count: int = 0
    # For conditional routing: True if first item has a source reference
    has_source_ref: bool = False
    status: str = "success"


class GetSourceContentInput(BaseModel):
    """Input for get_source_content_for_drafting node."""
    source_ref_id: Optional[str] = None


class SourceContent(BaseModel):
    """Source content for drafting."""
    content_raw: Optional[str] = None
    title: str = ""
    author: Optional[str] = None
    url: str = ""
    publication_handle: Optional[str] = None
    likes_count: int = 0


class GetSourceContentOutput(BaseModel):
    """Output from get_source_content_for_drafting node."""
    content: Optional[SourceContent] = None
    status: str = "success"


class LoadDraftingContextInput(BaseModel):
    """Input for load_drafting_context node."""
    source_ref_id: Optional[str] = None
    length_mode: str = "standard"
    custom_word_count: Optional[int] = None


class LoadDraftingContextOutput(BaseModel):
    """Output from load_drafting_context node."""
    source_content: Optional[str] = None  # Raw content from source article (if source-derived)
    target_word_count: int = 1250  # Calculated based on length_mode
    status: str = "success"


class UpdateQueueItemDraftInput(BaseModel):
    """Input for update_queue_item_draft node."""
    queue_id: str = ""
    draft_content: Optional[str] = None
    draft_image_url: Optional[str] = None
    draft_metadata: Optional[Dict[str, Any]] = None


class UpdateQueueItemDraftOutput(BaseModel):
    """Output from update_queue_item_draft node."""
    queue_id: str = ""
    status: str = "success"


class MarkQueueItemReviewedInput(BaseModel):
    """Input for mark_queue_item_reviewed node."""
    queue_id: str = ""
    approved: bool = True
    feedback: Optional[str] = None


class MarkQueueItemReviewedOutput(BaseModel):
    """Output from mark_queue_item_reviewed node."""
    queue_id: str = ""
    new_status: str = ""
    status: str = "success"


class GetQueueItemForPublishInput(BaseModel):
    """Input for get_queue_item_for_publish node."""
    queue_id: str = ""


class QueueItemForPublish(BaseModel):
    """A queue item ready for publishing."""
    id: str
    title: str
    draft_content: Optional[str] = None
    draft_image_url: Optional[str] = None


class GetQueueItemForPublishOutput(BaseModel):
    """Output from get_queue_item_for_publish node."""
    item: Optional[QueueItemForPublish] = None
    status: str = "success"


class FinalizePublicationInput(BaseModel):
    """Input for finalize_publication node."""
    target_handle: str = Field(default="", description="Target publication handle")
    queue_id: str = ""
    published_url: str = ""
    published_title: str = ""


class FinalizePublicationOutput(BaseModel):
    """Output from finalize_publication node."""
    target_item_id: Optional[str] = None
    published_url: str = ""
    status: str = "success"


class UpdateItemEmbeddingInput(BaseModel):
    """Input for update_item_embedding node."""
    item_type: str = Field(default="source", description="Type: 'source', 'target', or 'queue'")
    item_id: str = ""
    embedding: List[float] = Field(default_factory=list)


class UpdateItemEmbeddingOutput(BaseModel):
    """Output from update_item_embedding node."""
    item_id: str = ""
    status: str = "success"


class SetQueueItemDraftingInput(BaseModel):
    """Input for set_queue_item_drafting node."""
    queue_id: str = ""


class SetQueueItemDraftingOutput(BaseModel):
    """Output from set_queue_item_drafting node."""
    queue_id: str = ""
    new_status: str = ""
    status: str = "success"


class ResetDraftingStatusInput(BaseModel):
    """Input for reset_drafting_status node (error recovery)."""
    queue_id: Optional[str] = None  # If not provided, uses items[0].id from state


class ResetDraftingStatusOutput(BaseModel):
    """Output from reset_drafting_status node."""
    queue_id: Optional[str] = None
    reset: bool = False
    status: str = "success"


class ResetStaleDraftingItemsInput(BaseModel):
    """Input for reset_stale_drafting_items node (timeout-based safety net)."""
    target_handle: str = Field(description="Target publication handle")
    stale_minutes: int = Field(default=30, description="Reset items stuck in drafting longer than this")


class ResetStaleDraftingItemsOutput(BaseModel):
    """Output from reset_stale_drafting_items node."""
    reset_count: int = 0
    reset_ids: List[str] = Field(default_factory=list)
    status: str = "success"


class GetItemsWithoutEmbeddingsInput(BaseModel):
    """Input for get_items_without_embeddings node."""
    item_type: str = Field(default="source", description="Type: 'source', 'target', or 'queue'")
    limit: int = Field(default=100, description="Max items to return")


class ItemWithoutEmbedding(BaseModel):
    """An item that needs an embedding."""
    id: str
    text: str


class GetItemsWithoutEmbeddingsOutput(BaseModel):
    """Output from get_items_without_embeddings node."""
    items: List[ItemWithoutEmbedding] = Field(default_factory=list)
    count: int = 0
    status: str = "success"


# =============================================================================
# TARGET CONFIG SCHEMAS
# =============================================================================

class LoadTargetConfigInput(BaseModel):
    """Input for load_target_config node."""
    target_handle: str = Field(description="Target publication handle")


class LoadTargetConfigOutput(BaseModel):
    """Output from load_target_config node."""
    article_prompt: str = ""
    hero_image_prompt: str = ""
    inline_image_prompt: str = ""
    # Substack session cookies for authenticated scraping
    substack_cookies: Optional[Dict[str, str]] = None
    # Whitelist of publications where we have paid subscriptions
    # For these, we MUST get full content (workflow fails if cookies expire)
    paid_subscription_handles: List[str] = Field(default_factory=list)
    status: str = "success"


# =============================================================================
# IMAGE ANALYSIS SCHEMAS (Two-Pass Drafting)
# =============================================================================

class ImagePlacement(BaseModel):
    """Describes where a supplementary image should be placed."""
    after_section: str = Field(description="Section heading after which to insert image")
    prompt: str = Field(description="Image generation prompt (max 200 words)")
    alt_text: str = Field(description="Accessibility alt text for the image")
    rationale: str = Field(default="", description="Why this image adds value here")


class HeroImageSpec(BaseModel):
    """Specification for the hero/featured image."""
    prompt: str = Field(description="Image generation prompt (max 200 words)")
    alt_text: str = Field(description="Accessibility alt text for the image")


class ImageAnalysisResult(BaseModel):
    """Result of analyzing an article for image placement."""
    hero_image: HeroImageSpec
    supplementary_images: List[ImagePlacement] = Field(
        default_factory=list,
        description="0-2 supplementary images"
    )


class AnalyzeArticleForImagesInput(BaseModel):
    """Input for analyze_article_for_images node."""
    draft_content: str = Field(description="The article markdown content")
    title: str = Field(default="", description="Article title for context")
    min_supplementary: int = Field(default=0, description="Minimum supplementary images")
    max_supplementary: int = Field(default=2, description="Maximum supplementary images")
    # LLM config override (optional - falls back to workflow config, then .env)
    llm_config: Optional[LLMConfig] = Field(default=None, description="Override LLM provider/model")


class AnalyzeArticleForImagesOutput(BaseModel):
    """Output from analyze_article_for_images node."""
    image_analysis: Optional[ImageAnalysisResult] = None
    hero_prompt: str = Field(default="", description="Flattened hero prompt for convenience")
    supplementary_count: int = 0
    status: str = "success"


# =============================================================================
# BATCH IMAGE GENERATION SCHEMAS
# =============================================================================

class GeneratedImage(BaseModel):
    """Result of generating a single image."""
    id: str
    url: Optional[str] = None
    alt_text: str = ""
    placement: Optional[str] = None
    status: str = "success"
    error: Optional[str] = None


class GenerateAllImagesInput(BaseModel):
    """Input for generate_all_images node (batch generation)."""
    image_analysis: ImageAnalysisResult
    hero_image_style: Optional[str] = Field(default=None, description="Style to append to hero image prompts")
    inline_image_style: Optional[str] = Field(default=None, description="Style to append to inline image prompts")
    # Image config override (optional - falls back to .env)
    image_config: Optional[ImageConfig] = Field(default=None, description="Override image provider/model")


class GenerateAllImagesOutput(BaseModel):
    """Output from generate_all_images node."""
    hero_image: Optional[GeneratedImage] = None
    supplementary_images: List[GeneratedImage] = Field(default_factory=list)
    total_generated: int = 0
    total_failed: int = 0
    status: str = "success"


# =============================================================================
# STITCH DRAFT SCHEMAS
# =============================================================================

class StitchDraftInput(BaseModel):
    """Input for stitch_draft node."""
    draft_content: str = Field(description="Original article markdown")
    hero_image: Optional[GeneratedImage] = None
    supplementary_images: List[GeneratedImage] = Field(default_factory=list)


class StitchDraftOutput(BaseModel):
    """Output from stitch_draft node."""
    final_content: str = Field(default="", description="Article with images inserted at correct positions")
    hero_image_url: Optional[str] = None
    image_count: int = 0
    status: str = "success"


# =============================================================================
# GENERATE AND STITCH IMAGES SCHEMAS (Combined node for linear workflow)
# =============================================================================

class GenerateAndStitchImagesInput(BaseModel):
    """Input for generate_and_stitch_images node - combines generation and stitching."""
    draft_content: str = Field(description="Draft content with embedded image prompts")
    image_analysis: ImageAnalysisResult = Field(description="Image analysis with hero and supplementary specs")
    hero_image_style: Optional[str] = Field(default=None, description="Style to append to hero image prompts")
    inline_image_style: Optional[str] = Field(default=None, description="Style to append to inline image prompts")
    generate_images: bool = Field(default=False, description="If True, generate actual images; if False, pass through with prompts")
    # Image config override (optional - falls back to .env)
    image_config: Optional[ImageConfig] = Field(default=None, description="Override image provider/model")


class GenerateAndStitchImagesOutput(BaseModel):
    """Output from generate_and_stitch_images node."""
    final_content: str = Field(default="", description="Final content (with images if generated, or with prompts)")
    hero_image_url: Optional[str] = None
    image_count: int = 0
    images_generated: bool = False
    status: str = "success"


# =============================================================================
# DISCOVERY WORKFLOW SCHEMAS
# =============================================================================

class DiscoveredPost(BaseModel):
    """A post discovered during publication discovery."""
    publication_handle: str = ""
    url: str = ""
    title: str = ""
    subtitle: str = ""  # Merged subtitle/description - stores whichever is longer
    content_raw: Optional[str] = None  # Full post content (for vector embeddings)
    author: Optional[str] = None  # Author name
    substack_post_id: Optional[str] = None  # Substack's numeric post ID
    slug: Optional[str] = None  # URL slug
    tags: List[str] = Field(default_factory=list)  # Post tags from Substack (e.g., "System Design", "Coding Interviews")
    section_name: str = ""  # Section within the publication (e.g., "Deep Dives", "News Roundup")
    post_type: str = ""  # 'recent' or 'top' (discovery context)
    content_type: str = ""  # 'newsletter', 'podcast', 'thread' (Substack content type)
    audience: str = ""  # 'everyone', 'only_paid', 'founding'
    published_at: Optional[str] = None
    likes_count: int = 0
    comments_count: int = 0
    word_count: int = 0


class DiscoveredPublication(BaseModel):
    """A publication discovered from Substack leaderboard."""
    handle: str = ""
    name: str = ""
    description: str = ""
    subscriber_count: int = 0
    custom_domain: Optional[str] = None
    author_name: str = ""
    category: str = ""
    leaderboard_type: str = ""  # 'paid' or 'trending'
    leaderboard_rank: int = 0
    posts: List[DiscoveredPost] = Field(default_factory=list)

    @field_validator('subscriber_count', mode='before')
    @classmethod
    def parse_subscriber_count(cls, v):
        """
        Parse subscriber count from various formats.

        The Substack API sometimes returns strings like:
        - "Over 80,000 subscribers"
        - "100,000+ subscribers"
        - "1.2M subscribers"
        """
        if isinstance(v, int):
            return v
        if v is None:
            return 0
        if isinstance(v, str):
            import re
            # Remove commas and common text
            cleaned = v.lower().replace(',', '').replace('subscribers', '').replace('subscriber', '')
            cleaned = cleaned.replace('over', '').replace('+', '').strip()

            # Handle "1.2M" or "1.2K" format
            if 'm' in cleaned:
                cleaned = cleaned.replace('m', '')
                try:
                    return int(float(cleaned) * 1_000_000)
                except ValueError:
                    return 0
            if 'k' in cleaned:
                cleaned = cleaned.replace('k', '')
                try:
                    return int(float(cleaned) * 1_000)
                except ValueError:
                    return 0

            # Extract digits
            digits = re.sub(r'[^\d]', '', cleaned)
            if digits:
                return int(digits)
            return 0
        return 0


class ScrapeLeaderboardsInput(BaseModel):
    """Input for scrape_leaderboards node."""
    category: str = Field(description="Substack category (e.g., 'technology')")


class ScrapeLeaderboardsOutput(BaseModel):
    """Output from scrape_leaderboards node."""
    publications: List[DiscoveredPublication] = Field(default_factory=list)
    total_count: int = 0
    status: str = "success"


class FetchPublicationPostsInput(BaseModel):
    """Input for fetch_publication_posts node."""
    publications: List[DiscoveredPublication] = Field(default_factory=list)
    recent_count: int = 3
    top_count: int = 3


class FetchPublicationPostsOutput(BaseModel):
    """Output from fetch_publication_posts node."""
    publications: List[DiscoveredPublication] = Field(default_factory=list)
    total_posts: int = 0
    status: str = "success"


class ScoreKeywordRelevanceInput(BaseModel):
    """Input for score_keyword_relevance node."""
    publications: List[DiscoveredPublication] = Field(default_factory=list)
    keywords: str = Field(description="Semicolon-separated keywords")
    top_n: int = 50


class ScoreKeywordRelevanceOutput(BaseModel):
    """Output from score_keyword_relevance node."""
    filtered_publications: List[DiscoveredPublication] = Field(default_factory=list)
    status: str = "success"


class ScoreLLMRelevanceInput(BaseModel):
    """Input for score_llm_relevance node."""
    publications: List[DiscoveredPublication] = Field(default_factory=list)
    keywords: str = ""
    top_n: int = 10
    llm_config: Optional[LLMConfig] = None


class ScoreLLMRelevanceOutput(BaseModel):
    """Output from score_llm_relevance node."""
    scored_publications: List[DiscoveredPublication] = Field(default_factory=list)
    status: str = "success"


class GenerateDiscoveryReportInput(BaseModel):
    """Input for generate_discovery_report node."""
    publications: List[DiscoveredPublication] = Field(default_factory=list)
    scored_publications: List[DiscoveredPublication] = Field(default_factory=list)
    filtered_publications: List[DiscoveredPublication] = Field(default_factory=list)
    top_n: int = 10
    category: str = ""
    keywords: str = ""


class GenerateDiscoveryReportOutput(BaseModel):
    """Output from generate_discovery_report node."""
    report_title: str = ""
    report_markdown: str = ""
    report_tags: List[str] = Field(default_factory=list)
    report_json: Dict[str, Any] = Field(default_factory=dict)
    status: str = "success"


class ScoredPublicationResult(BaseModel):
    """A publication scored by Borda ranking with 4 signals."""
    handle: str = ""
    name: str = ""
    subscriber_count: int = 0
    matching_posts: int = 0
    avg_similarity: float = 0.0
    max_similarity: float = 0.0
    total_likes: int = 0
    leaderboard_type: str = ""  # 'paid' or 'trending'
    leaderboard_rank: int = 0  # rank on the leaderboard (lower = better)
    # Borda ranking fields (lower = better)
    borda_score: int = 0  # sum of ranks across 4 signals
    rank_by_paid: Optional[int] = None  # rank among paid leaderboard entries (None if trending)
    rank_by_trending: Optional[int] = None  # rank among trending leaderboard entries (None if paid)
    rank_by_avg_relevance: int = 0  # rank by average similarity
    rank_by_max_relevance: int = 0  # rank by max similarity (best post)
    selection_reason: str = ""


class RankPublicationsOutput(BaseModel):
    """Output from rank_publications node - all publications ranked by Borda score."""
    ranked_publications: List[ScoredPublicationResult] = Field(default_factory=list)
    total_in_category: int = 0
    status: str = "success"


class GenerateReportInput(BaseModel):
    """Input for generate_report node."""
    ranked_publications: List[ScoredPublicationResult] = Field(default_factory=list)
    top_n: int = Field(default=10, description="Number of publications to include in report")
    category: str = ""
    keywords: str = ""
    kv_key: str = Field(default="discovery_sources", description="KV store key to save results under")


class GenerateReportOutput(BaseModel):
    """Output from generate_report node."""
    report_title: str = ""
    report_markdown: str = ""
    report_tags: List[str] = Field(default_factory=list)
    report_json: Dict[str, Any] = Field(default_factory=dict)
    status: str = "success"


# =============================================================================
# LLM-BASED RELEVANCE SCORING SCHEMAS
# =============================================================================

class ScorePostsLLMInput(BaseModel):
    """Input for score_posts_llm node - LLM scores all posts for relevance."""
    publications: List[DiscoveredPublication] = Field(default_factory=list)
    keywords: str = Field(default="", description="Keywords to score relevance against")
    scoring_prompt: str = Field(default="", description="Custom prompt for LLM scoring")
    llm_config: Optional[LLMConfig] = None
    llm_retries: int = Field(default=1, description="Number of retries for LLM timeout/network errors (0 = no retries)")


class PostScore(BaseModel):
    """LLM relevance score for a single post."""
    url: str = ""
    score: int = 0  # 0-100
    reason: str = ""


class ScoredPublication(BaseModel):
    """A publication with LLM-scored posts."""
    handle: str = ""
    name: str = ""
    subscriber_count: int = 0
    leaderboard_type: str = ""
    leaderboard_rank: int = 0
    post_scores: List[PostScore] = Field(default_factory=list)
    avg_score: float = 0.0
    max_score: int = 0
    total_likes: int = 0


class ScorePostsLLMOutput(BaseModel):
    """Output from score_posts_llm node."""
    scored_publications: List[ScoredPublication] = Field(default_factory=list)
    total_posts_scored: int = 0
    status: str = "success"


class RankPublicationsLLMInput(BaseModel):
    """Input for rank_publications_llm node - Borda ranking with LLM scores."""
    scored_publications: List[ScoredPublication] = Field(default_factory=list)


class RankedPublication(BaseModel):
    """A publication with Borda ranking score."""
    handle: str = ""
    name: str = ""
    subscriber_count: int = 0
    leaderboard_type: str = ""
    paid_rank: int = 0
    trending_rank: int = 0
    avg_llm_score: float = 0.0
    borda_score: int = 0  # Lower is better
    total_likes: int = 0


class RankPublicationsLLMOutput(BaseModel):
    """Output from rank_publications_llm node."""
    ranked_publications: List[RankedPublication] = Field(default_factory=list)
    total_count: int = 0
    status: str = "success"


# =============================================================================
# UNIFIED POSTS SCHEMAS
# =============================================================================

class PostData(BaseModel):
    """A post for the unified posts table."""
    url: str
    publication_handle: str
    title: str
    subtitle: Optional[str] = None  # Merged subtitle/description - stores whichever is longer
    content_raw: Optional[str] = None
    author: Optional[str] = None
    substack_post_id: Optional[str] = None
    slug: Optional[str] = None
    post_type: Optional[str] = None  # 'newsletter', 'podcast', 'thread'
    audience: Optional[str] = None   # 'everyone', 'only_paid', 'founding'
    word_count: Optional[int] = None
    tags: List[str] = Field(default_factory=list)  # Post tags (e.g., "System Design")
    section_name: Optional[str] = None  # Section within publication (e.g., "Deep Dives")
    is_paywalled: bool = False
    has_full_content: bool = True
    likes_count: int = 0
    comments_count: int = 0
    shares_count: int = 0
    published_at: Optional[str] = None
    discovery_source: Optional[str] = None  # 'content_sync', 'publication_discovery'


class UpsertPostsInput(BaseModel):
    """Input for upsert_posts node."""
    posts: List[PostData] = Field(default_factory=list)
    discovery_source: str = Field(default="content_sync", description="Source of these posts")


class UpsertedPost(BaseModel):
    """A post that was inserted or updated."""
    id: str
    url: str
    title: str
    publication_handle: str
    is_new: bool = True  # True if inserted, False if updated


class UpsertPostsOutput(BaseModel):
    """Output from upsert_posts node."""
    inserted_count: int = 0
    updated_count: int = 0
    upserted_posts: List[UpsertedPost] = Field(default_factory=list)
    status: str = "success"


class GetPostsInput(BaseModel):
    """Input for get_posts node."""
    publication_handles: List[str] = Field(default_factory=list, description="Filter by publication handles")
    min_likes: int = Field(default=0, description="Minimum likes threshold")
    max_age_days: Optional[int] = Field(default=None, description="Max age in days")
    has_content: Optional[bool] = Field(default=None, description="Filter by content availability")
    has_embedding: Optional[bool] = Field(default=None, description="Filter by embedding availability")
    limit: int = Field(default=100, description="Max posts to return")
    offset: int = Field(default=0, description="Offset for pagination")
    order_by: str = Field(default="likes_count", description="Order by: likes_count, published_at")


class PostResult(BaseModel):
    """A post from the database."""
    id: str
    url: str
    publication_handle: str
    title: str
    subtitle: Optional[str] = None
    author: Optional[str] = None
    likes_count: int = 0
    comments_count: int = 0
    shares_count: int = 0
    published_at: Optional[str] = None
    has_content: bool = False
    has_embedding: bool = False
    is_paywalled: bool = False
    has_full_content: bool = True


class GetPostsOutput(BaseModel):
    """Output from get_posts node."""
    posts: List[PostResult] = Field(default_factory=list)
    count: int = 0
    total_count: int = 0  # Total matching (before limit)
    status: str = "success"


class GetPostsNeedingContentInput(BaseModel):
    """Input for get_posts_needing_content node."""
    publication_handles: List[str] = Field(default_factory=list, description="Filter by publication handles")
    limit: int = Field(default=100, description="Max posts to return")


class GetPostsNeedingContentOutput(BaseModel):
    """Output from get_posts_needing_content node."""
    posts: List[PostResult] = Field(default_factory=list)
    count: int = 0
    status: str = "success"


class UpdatePostMetricsInput(BaseModel):
    """Input for update_post_metrics node."""
    updates: List[Dict[str, Any]] = Field(default_factory=list, description="List of {url, likes_count, comments_count, shares_count}")


class UpdatePostMetricsOutput(BaseModel):
    """Output from update_post_metrics node."""
    updated_count: int = 0
    status: str = "success"


class IdentifyNewPostUrlsInput(BaseModel):
    """Input for identify_new_post_urls node - checks which URLs are new."""
    urls: List[str] = Field(default_factory=list, description="URLs to check")
    posts: List[Dict[str, Any]] = Field(default_factory=list, description="Full post objects with url key")


class IdentifyNewPostUrlsOutput(BaseModel):
    """Output from identify_new_post_urls node."""
    new_urls: List[str] = Field(default_factory=list)
    existing_urls: List[str] = Field(default_factory=list)
    new_posts: List[Dict[str, Any]] = Field(default_factory=list, description="Full post objects for new URLs")
    status: str = "success"


class GetPostsNeedingEmbeddingsInput(BaseModel):
    """Input for get_posts_needing_embeddings node."""
    publication_handles: List[str] = Field(default_factory=list, description="Filter by publication handles")
    limit: int = Field(default=500, description="Max posts to return")


class PostNeedingEmbedding(BaseModel):
    """A post that needs an embedding generated."""
    id: str
    url: str
    title: str
    content_raw: Optional[str] = None


class GetPostsNeedingEmbeddingsOutput(BaseModel):
    """Output from get_posts_needing_embeddings node."""
    posts: List[PostNeedingEmbedding] = Field(default_factory=list)
    count: int = 0
    status: str = "success"


class UpdatePostEmbeddingsInput(BaseModel):
    """Input for update_post_embeddings node."""
    items: List[Dict[str, Any]] = Field(default_factory=list, description="List of {id, embedding}")


class UpdatePostEmbeddingsOutput(BaseModel):
    """Output from update_post_embeddings node."""
    updated_count: int = 0
    status: str = "success"


class QueueSourcesFromPostsInput(BaseModel):
    """Input for queue_sources_from_posts node - queues from unified posts table."""
    source_handles: List[str] = Field(default_factory=list, description="Source publication handles to draw from")
    target_handle: str = Field(default="", description="Target publication handle (for production_queue)")
    count: int = Field(default=10, description="Number of articles to queue")
    min_likes: int = Field(default=10, description="Minimum likes threshold")
    max_age_days: int = Field(default=90, description="Only consider articles published within this many days")
    recent_days: int = Field(default=7, description="Articles within this many days are considered 'recent'")


class QueueSourcesFromPostsOutput(BaseModel):
    """Output from queue_sources_from_posts node."""
    queued_count: int = 0
    queued_from_recent: int = 0
    queued_from_top: int = 0
    queued_items: List[QueuedSourceItem] = Field(default_factory=list)
    status: str = "success"
