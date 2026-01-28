"""
LLM node functions for the AI Substack Mirroring Engine.

These workflow nodes handle AI-powered content generation:
- generate_topic_ideas: Gap analysis - generate new topic ideas
- draft_article: Generate full article draft
- generate_image_prompt: Create image generation prompts
"""
import os
import hashlib
import structlog
import httpx
import json

from pathlib import Path
from typing import Dict, Optional, TypeVar, Type, List, Any
from pydantic import BaseModel, ValidationError, field_validator, Field
from sqlalchemy import select

from .prompts import (
    GENERATE_IDEAS_PROMPT,
    GENERATE_TOPIC_IDEAS_PROMPT,
    DEDUP_IDEAS_PROMPT,
    ANALYZE_IMAGES_PROMPT,
    INJECT_IMAGES_PROMPT,
)
from .schemas import (
    LLMConfig, resolve_model,
    GenerateTopicIdeasInput, GenerateTopicIdeasOutput, TopicIdea,
    DraftArticleInput, DraftArticleOutput,
    GenerateImagePromptInput, GenerateImagePromptOutput,
    GenerateImageInput, GenerateImageOutput,
    RefineDraftInput, RefineDraftOutput,
    EmbedImagePromptsInput, EmbedImagePromptsOutput,
    # Image analysis schemas
    AnalyzeArticleForImagesInput, AnalyzeArticleForImagesOutput,
    ImageAnalysisResult, HeroImageSpec, ImagePlacement,
    GenerateAllImagesInput, GenerateAllImagesOutput, GeneratedImage,
    StitchDraftInput, StitchDraftOutput,
    # Combined image generation + stitch (linear workflow)
    GenerateAndStitchImagesInput, GenerateAndStitchImagesOutput,
    # Placeholder injection schemas (legacy)
    InjectPlaceholdersInput, InjectPlaceholdersOutput, ImagePromptOutput,
    # New unified image injection schemas (v5.3+)
    InjectImagesInput, InjectImagesOutput, LLMInjectImagesResponse, CompiledImagePrompt,
    # Image generation from prompts
    GenerateImagesFromPromptsInput, GenerateImagesFromPromptsOutput, GeneratedImageResult,
    # Idea generation schemas
    GenerateIdeasFromSourcesInput, GenerateIdeasFromSourcesOutput, GeneratedIdea,
    # Dedup schemas
    DedupIdeasLLMInput, DedupIdeasLLMOutput, DedupResult,
)

logger = structlog.get_logger()

# Type variable for generic Pydantic model validation
T = TypeVar("T", bound=BaseModel)


# =============================================================================
# DEV CACHE (for speeding up development iteration)
# =============================================================================
# Enable with LLM_DEV_CACHE=true in .env
# Clear specific node: rm /tmp/llm_dev_cache/<node_name>_*.json
# Clear all: rm -rf /tmp/llm_dev_cache/

LLM_DEV_CACHE_DIR = Path("/tmp/llm_dev_cache")


def _is_dev_cache_enabled() -> bool:
    """Check if LLM dev cache is enabled via env var."""
    return os.environ.get("LLM_DEV_CACHE", "").lower() in ("true", "1", "yes")


def _get_cache_key(node_name: str, data: Any) -> str:
    """Generate a cache key from node name and input data."""
    data_str = json.dumps(data, sort_keys=True, default=str)
    hash_val = hashlib.sha256(data_str.encode()).hexdigest()[:16]
    return f"{node_name}_{hash_val}"


def _get_cached_response(node_name: str, data: Any) -> Optional[dict]:
    """Get cached LLM response if exists and dev cache is enabled."""
    if not _is_dev_cache_enabled():
        return None

    cache_key = _get_cache_key(node_name, data)
    cache_file = LLM_DEV_CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cached = json.load(f)
                logger.info("llm_dev_cache_hit", node=node_name, cache_key=cache_key)
                return cached
        except Exception as e:
            logger.warning("llm_dev_cache_read_error", error=str(e))

    return None


def _save_to_cache(node_name: str, data: Any, response: dict) -> None:
    """Save LLM response to dev cache."""
    if not _is_dev_cache_enabled():
        return

    try:
        LLM_DEV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_key = _get_cache_key(node_name, data)
        cache_file = LLM_DEV_CACHE_DIR / f"{cache_key}.json"

        with open(cache_file, "w") as f:
            json.dump(response, f, indent=2, default=str)

        logger.info("llm_dev_cache_saved", node=node_name, cache_key=cache_key)
    except Exception as e:
        logger.warning("llm_dev_cache_write_error", error=str(e))


# =============================================================================
# PYDANTIC-VALIDATED LLM CALLS
# =============================================================================

class LLMScoresResponse(BaseModel):
    """Expected response format for LLM scoring calls (simple array)."""
    scores: List[int]


class PublicationScoreItem(BaseModel):
    """Single publication score from LLM."""
    handle: str
    score: int
    reason: str = ""


class LLMPublicationScoresResponse(BaseModel):
    """Expected response format for LLM publication relevance scoring."""
    scores: List[PublicationScoreItem]


class LLMGeneratedIdea(BaseModel):
    """Single idea from LLM response."""
    title: str
    summary: str
    post_type: str  # "paid" or "free"
    inspiration: str
    source_posts: List[int]  # Numbers referencing source posts (1-indexed from prompt)

    @field_validator('post_type', mode='before')
    @classmethod
    def normalize_post_type(cls, v):
        """Normalize post_type to 'free' or 'paid'."""
        if isinstance(v, str):
            v_lower = v.lower()
            if 'paid' in v_lower:
                return 'paid'
            if 'free' in v_lower:
                return 'free'
        return v

    @field_validator('source_posts', mode='before')
    @classmethod
    def ensure_list(cls, v):
        """Ensure source_posts is a list of ints."""
        if v is None:
            return []
        if isinstance(v, int):
            return [v]
        return v


class LLMIdeasResponse(BaseModel):
    """Expected response format for idea generation."""
    ideas: List[LLMGeneratedIdea]


class LLMTopicIdea(BaseModel):
    """Single topic idea from LLM response."""
    title: str
    rationale: str


class LLMTopicIdeasResponse(BaseModel):
    """Expected response format for topic idea generation."""
    ideas: List[LLMTopicIdea]


class LLMHeroImage(BaseModel):
    """Hero image spec from LLM."""
    prompt: str
    alt_text: str = "Article hero image"


class LLMSupplementaryImage(BaseModel):
    """Supplementary image placement from LLM."""
    after_section: str
    prompt: str
    alt_text: str = ""
    rationale: str = ""


class LLMImageAnalysisResponse(BaseModel):
    """Expected response format for image analysis."""
    hero_image: LLMHeroImage
    supplementary_images: List[LLMSupplementaryImage] = Field(default_factory=list)


class LLMDedupIdea(BaseModel):
    """Single idea dedup result from LLM."""
    title: str
    similarity_score: int = Field(default=0, ge=0, le=100, description="0-100 similarity score")
    is_duplicate: bool
    similar_to: Optional[str] = None  # Title of existing content it duplicates
    reason: str  # Brief explanation of decision


class LLMDedupResponse(BaseModel):
    """Expected response format for idea deduplication."""
    results: List[LLMDedupIdea]


async def call_llm_validated(
    prompt: str,
    config: dict,
    response_model: Type[T],
    max_tokens: int = 2000,
    max_retries: int = 2,
) -> T:
    """
    Call LLM with Pydantic validation and retry on validation failure.

    If LLM returns invalid JSON or JSON that doesn't match the schema,
    we retry with the validation error appended to the prompt so LLM
    can correct itself.

    Dev cache: Set LLM_DEV_CACHE=true in .env to cache responses.
    Clear cache: rm /tmp/llm_dev_cache/*.json

    Args:
        prompt: The prompt to send to LLM
        config: LLM configuration dict (provider, model, keys)
        response_model: Pydantic model class to validate response against
        max_tokens: Max tokens for LLM response
        max_retries: Number of retries on validation failure (default 2)

    Returns:
        Validated Pydantic model instance

    Raises:
        RuntimeError: If all retries exhausted or LLM call fails
    """
    # Check dev cache
    model_name = config.get("model", "unknown")
    cache_key_data = {"prompt": prompt, "model": model_name}
    cached = _get_cached_response("llm", cache_key_data)
    if cached:
        try:
            return response_model.model_validate(cached)
        except ValidationError as e:
            # Cache exists but doesn't match expected schema - call LLM fresh
            logger.warning("llm_dev_cache_schema_mismatch", error=str(e)[:200])

    current_prompt = prompt
    last_error = None

    # Get JSON schema from Pydantic model for providers that support it (e.g., Gemini)
    pydantic_schema = response_model.model_json_schema()

    for attempt in range(max_retries + 1):
        try:
            # Call LLM with JSON mode and schema (schema used by Gemini for structured output)
            response = await _call_llm(
                current_prompt,
                config,
                max_tokens=max_tokens,
                json_mode=True,
                response_schema=pydantic_schema,
                temperature=config.get("temperature"),
            )

            # Try to parse and validate with Pydantic
            try:
                validated = response_model.model_validate_json(response)
                if attempt > 0:
                    logger.info("llm_validation_retry_succeeded", attempt=attempt + 1)
                # Save to dev cache
                _save_to_cache("llm", cache_key_data, validated.model_dump())
                return validated

            except ValidationError as e:
                last_error = e
                error_details = e.json()

                if attempt < max_retries:
                    # Build retry prompt with error feedback
                    logger.warning(
                        "llm_validation_failed_retrying",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        error=str(e),
                    )
                    current_prompt = f"""{prompt}

IMPORTANT: Your previous response was invalid. Please fix the following validation errors and try again:

{error_details}

Respond with valid JSON that matches the expected schema."""
                else:
                    # All retries exhausted
                    logger.error(
                        "llm_validation_failed_exhausted",
                        attempts=max_retries + 1,
                        error=str(e),
                        response_preview=response[:500] if response else "EMPTY",
                    )
                    raise RuntimeError(
                        f"LLM response validation failed after {max_retries + 1} attempts: {e}"
                    ) from e

        except json.JSONDecodeError as e:
            last_error = e
            if attempt < max_retries:
                logger.warning(
                    "llm_json_parse_failed_retrying",
                    attempt=attempt + 1,
                    error=str(e),
                )
                current_prompt = f"""{prompt}

IMPORTANT: Your previous response was not valid JSON. Error: {e}

Please respond with valid JSON only, no additional text or markdown formatting."""
            else:
                raise RuntimeError(f"LLM returned invalid JSON after {max_retries + 1} attempts: {e}") from e

    # Should not reach here, but just in case
    raise RuntimeError(f"LLM validation failed: {last_error}")


def _get_llm_config(ctx, llm_config: Optional[LLMConfig] = None) -> dict:
    """
    Get LLM configuration with cascading priority.

    Resolution order (first non-None wins):
    1. Node params (llm_config.model - user-friendly name like "Gemini 3 Pro")
    2. Environment variables (.env: LLM_PROVIDER + LLM_MODEL)
    3. Defaults (ollama/llama3.1)

    Args:
        ctx: Execution context with access to secrets
        llm_config: Optional override from node params (can be dict or LLMConfig)

    Returns:
        dict with provider, model, temperature, and API keys
    """
    provider = None
    model = None
    temperature = None

    # 1. Try model field from node params (e.g., "Gemini 3 Pro")
    # Handle both dict and Pydantic model
    if llm_config:
        model_value = llm_config.get("model") if isinstance(llm_config, dict) else llm_config.model
        if model_value:
            resolved_provider, resolved_model = resolve_model(model_value)
            if resolved_provider:
                provider = resolved_provider
                model = resolved_model
        # Get temperature from config (can be 0, so check for None explicitly)
        temp_value = llm_config.get("temperature") if isinstance(llm_config, dict) else llm_config.temperature
        if temp_value is not None:
            temperature = temp_value

    # 2. Fall back to environment variables
    if not provider:
        provider = ctx.get_secret("LLM_PROVIDER") or "ollama"
    if not model:
        model = ctx.get_secret("LLM_MODEL") or "llama3.1"

    return {
        "provider": provider,
        "model": model,
        "temperature": temperature,  # None means use provider default
        "ollama_host": ctx.get_secret("OLLAMA_HOST") or "http://localhost:11434",
        "openai_api_key": ctx.get_secret("OPENAI_API_KEY"),
        "anthropic_api_key": ctx.get_secret("ANTHROPIC_API_KEY"),
        "google_api_key": ctx.get_secret("GOOGLE_API_KEY"),
    }


async def generate_topic_ideas(
    ctx,
    params: GenerateTopicIdeasInput,
) -> GenerateTopicIdeasOutput:
    """
    Generate new topic ideas based on source patterns (Gap Analysis).

    Generates 2x requested count to provide buffer for filtering step.
    Existing topics are passed to LLM to avoid duplicates upfront.
    Returns all generated ideas - filtering step will cap at requested count.
    """
    config = _get_llm_config(ctx, params.llm_config)
    requested_count = params.count
    source_patterns = params.source_patterns
    existing_topics = params.existing_topics or []

    # Generate 2x to have buffer after filtering
    generate_count = requested_count * 2

    ctx.report_input({
        "requested_count": requested_count,
        "generate_count": generate_count,
        "source_patterns_count": len(source_patterns),
        "existing_topics_count": len(existing_topics),
        "provider": config["provider"],
        "model": config["model"],
        "existing_topics_to_avoid": existing_topics[:10] if existing_topics else [],  # Show first 10
    })

    # Build context from source patterns (token-efficient: title + likes only)
    pattern_lines = []
    if source_patterns:
        for p in source_patterns[:10]:
            # Handle both dict and object cases
            if isinstance(p, dict):
                title = p.get("title", "")
                likes = p.get("likes_count", 0)
            else:
                title = p.title
                likes = p.likes_count
            pattern_lines.append(f"- {title} ({likes} likes)")
    pattern_context = "\n".join(pattern_lines) if pattern_lines else "(No source patterns available)"

    # Build existing topics context (just titles to avoid)
    existing_lines = []
    if existing_topics:
        for topic in existing_topics[:50]:
            existing_lines.append(f"- {topic}")
    existing_context = "\n".join(existing_lines) if existing_lines else "(None)"

    # Generate response schema from Pydantic model
    response_schema = f"Respond with JSON matching this schema:\n```json\n{json.dumps(LLMTopicIdeasResponse.model_json_schema(), indent=2)}\n```"

    prompt = GENERATE_TOPIC_IDEAS_PROMPT.format(
        pattern_context=pattern_context,
        existing_context=existing_context,
        generate_count=generate_count,
        response_schema=response_schema,
    )

    try:
        validated_response = await call_llm_validated(
            prompt=prompt,
            config=config,
            response_model=LLMTopicIdeasResponse,
            max_tokens=2000,
        )

        ideas = [
            TopicIdea(
                title=i.title,
                rationale=i.rationale,
            )
            for i in validated_response.ideas
        ]

        logger.info(
            "topic_ideas_generated",
            generated=len(ideas),
            requested=requested_count,
        )

        ctx.report_output({
            "prompt": prompt,
            "topic_ideas": [
                {"title": i.title, "rationale": i.rationale}
                for i in ideas
            ],
            "count": len(ideas),
            "requested": requested_count,
            "status": "success",
        })

        return GenerateTopicIdeasOutput(
            topic_ideas=ideas,
            count=len(ideas),
            status="success",
        )

    except Exception as e:
        logger.error("topic_generation_failed", error=str(e))
        ctx.report_output({
            "status": "error",
            "error": str(e),
        })
        return GenerateTopicIdeasOutput(status="error")


async def generate_ideas_from_sources(
    ctx,
    params: GenerateIdeasFromSourcesInput,
) -> GenerateIdeasFromSourcesOutput:
    """
    Generate topic ideas based on top-performing source posts and my voice.

    Takes:
    - Top posts from source publications (what's performing well)
    - My published posts (for voice/style context)

    Generates ideas that tap into proven demand while fitting my voice.
    Returns both paid and free post ideas.
    """
    source_posts = params.source_posts
    my_posts = params.my_posts
    idea_count = params.idea_count
    llm_config = params.llm_config

    config = _get_llm_config(ctx, llm_config)

    ctx.report_input({
        "source_posts_count": len(source_posts),
        "my_posts_count": len(my_posts),
        "idea_count": idea_count,
        "provider": config["provider"],
        "model": config["model"],
    })

    if not source_posts:
        ctx.report_output({
            "status": "no_source_posts",
            "message": "No source posts available to generate ideas from. This can happen when: (1) No source publications configured, (2) No posts within the date range, or (3) All posts have been worked on already.",
            "source_posts_count": 0,
            "my_posts_count": len(my_posts) if my_posts else 0,
        })
        return GenerateIdeasFromSourcesOutput(status="no_source_posts")

    # Build source posts context (data only - header is in template)
    # Also build a mapping from post number (1-indexed) to post ID for later
    source_lines = []
    post_number_to_id: Dict[int, str] = {}  # 1 -> "uuid-123", 2 -> "uuid-456", etc.

    for i, post in enumerate(source_posts[:30], 1):  # Limit to top 30
        if isinstance(post, dict):
            title = post.get("title", "")
            likes = post.get("likes_count", 0)
            source = post.get("publication_handle", "")
            post_id = post.get("id", "")
        else:
            title = post.title
            likes = post.likes_count
            source = post.publication_handle
            post_id = getattr(post, "id", "")
        source_lines.append(f"{i}. \"{title}\" - {likes} likes (from @{source})")
        if post_id:
            post_number_to_id[i] = str(post_id)
    source_context = "\n".join(source_lines)

    # Build voice context (data only - header is in template)
    voice_lines = []
    if my_posts:
        for i, post in enumerate(my_posts[:10], 1):  # Limit to 10 posts
            if isinstance(post, dict):
                title = post.get("title", "")
                preview = post.get("content_preview", "")
                likes = post.get("likes_count", 0)
            else:
                title = post.title
                preview = post.content_preview
                likes = post.likes_count
            voice_lines.append(f"**{i}. {title}** ({likes} likes)\n{preview[:400]}...")
    my_voice_context = "\n\n".join(voice_lines) if voice_lines else "(No published posts available for voice context)"

    # Generate response schema from Pydantic model
    response_schema = f"Respond with JSON matching this schema:\n```json\n{json.dumps(LLMIdeasResponse.model_json_schema(), indent=2)}\n```"

    prompt = GENERATE_IDEAS_PROMPT.format(
        source_context=source_context,
        my_voice_context=my_voice_context,
        idea_count=idea_count,
        response_schema=response_schema,
    )

    try:
        # Use validated LLM call with automatic retry on schema errors
        validated_response = await call_llm_validated(
            prompt=prompt,
            config=config,
            response_model=LLMIdeasResponse,
            max_tokens=4000,
        )

        # Convert to output schema, mapping post numbers to IDs
        ideas = []
        for idea in validated_response.ideas[:idea_count]:
            # Convert source_posts numbers (1-indexed) to post IDs
            source_ids = []
            for num in idea.source_posts:
                if num in post_number_to_id:
                    source_ids.append(post_number_to_id[num])
                else:
                    logger.warning("invalid_source_post_number", number=num, max=len(post_number_to_id))

            ideas.append(GeneratedIdea(
                title=idea.title,
                summary=idea.summary,
                post_type=idea.post_type,
                inspiration_source=idea.inspiration,
                source_post_ids=source_ids,
            ))

        logger.info(
            "ideas_from_sources_generated",
            count=len(ideas),
            requested=idea_count,
        )

        ctx.report_output({
            "full_prompt": prompt,
            "ideas": [
                {
                    "title": i.title,
                    "summary": i.summary,
                    "post_type": i.post_type,
                    "inspiration": i.inspiration_source,
                    "source_post_ids": i.source_post_ids,
                }
                for i in ideas
            ],
            "post_number_to_id": post_number_to_id,  # For debugging
            "count": len(ideas),
            "status": "success",
        })

        return GenerateIdeasFromSourcesOutput(
            ideas=ideas,
            count=len(ideas),
            status="success",
        )

    except Exception as e:
        logger.error("ideas_from_sources_failed", error=str(e))
        ctx.report_output({
            "status": "error",
            "error": str(e),
        })
        raise RuntimeError(f"Failed to generate ideas from sources: {e}") from e


async def dedup_ideas_llm(
    ctx,
    params: DedupIdeasLLMInput,
) -> DedupIdeasLLMOutput:
    """
    Deduplicate ideas using LLM semantic understanding.

    Compares new idea titles against existing queue/published titles.
    More accurate than vector similarity for catching semantic duplicates.
    """
    from shared.database import get_db_session
    from shared.models import ProductionQueueItem, TargetItem

    target_handle = params.target_handle
    ideas = params.ideas
    llm_config = params.llm_config

    config = _get_llm_config(ctx, llm_config)

    ctx.report_input({
        "target_handle": target_handle,
        "ideas_count": len(ideas),
        "provider": config["provider"],
        "model": config["model"],
        # Long array last to avoid truncation of important fields
        "ideas": [
            {
                "title": idea.get("title", ""),
                "summary": idea.get("summary", ""),
                "post_type": idea.get("post_type", "free"),
                "inspiration": idea.get("inspiration", ""),
            }
            for idea in ideas
        ],
    })

    if not ideas:
        ctx.report_output({
            "status": "success",
            "message": "No ideas to dedup",
            "surviving_count": 0,
            "discarded_count": 0,
        })
        return DedupIdeasLLMOutput(status="success")

    # Fetch existing titles from queue and published
    async with get_db_session() as db:
        # Queue titles
        queue_result = await db.execute(
            select(ProductionQueueItem.title)
            .where(ProductionQueueItem.target_publication == target_handle)
        )
        queue_titles = [row[0] for row in queue_result.fetchall() if row[0]]

        # Published titles
        published_result = await db.execute(
            select(TargetItem.title)
            .where(TargetItem.target_publication == target_handle)
        )
        published_titles = [row[0] for row in published_result.fetchall() if row[0]]

    existing_titles = queue_titles + published_titles

    # If no existing content, all ideas survive
    if not existing_titles:
        ctx.report_output({
            "status": "success",
            "message": f"No existing content to compare against - all {len(ideas)} ideas survive",
            "surviving_count": len(ideas),
            "discarded_count": 0,
            "surviving_ideas": [
                {"title": i.get("title", ""), "post_type": i.get("post_type", "free")}
                for i in ideas
            ],
        })
        return DedupIdeasLLMOutput(
            surviving_ideas=ideas,
            surviving_count=len(ideas),
            status="success",
        )

    # Build compact inputs for LLM (titles only)
    existing_lines = [f"- {t}" for t in existing_titles[:100]]  # Limit to 100
    new_lines = [f"{i+1}. {idea.get('title', '')}" for i, idea in enumerate(ideas)]

    response_schema = f"Respond with JSON matching this schema:\n```json\n{json.dumps(LLMDedupResponse.model_json_schema(), indent=2)}\n```"

    prompt = DEDUP_IDEAS_PROMPT.format(
        existing_titles="\n".join(existing_lines),
        new_ideas="\n".join(new_lines),
        response_schema=response_schema,
    )

    try:
        validated_response = await call_llm_validated(
            prompt=prompt,
            config=config,
            response_model=LLMDedupResponse,
            max_tokens=1500,
        )

        # Process results
        surviving_ideas = []
        discarded_ideas = []

        # Build lookup by title
        ideas_by_title = {idea.get("title", ""): idea for idea in ideas}

        for result in validated_response.results:
            if result.is_duplicate:
                discarded_ideas.append(DedupResult(
                    title=result.title,
                    similarity_score=result.similarity_score,
                    is_duplicate=True,
                    similar_to=result.similar_to,
                    reason=result.reason,
                ))
            else:
                # Find original idea and keep it
                original_idea = ideas_by_title.get(result.title)
                if original_idea:
                    # Add reason and score to surviving idea
                    original_idea["dedup_reason"] = result.reason
                    original_idea["similarity_score"] = result.similarity_score
                    surviving_ideas.append(original_idea)

        logger.info(
            "ideas_deduped_llm",
            total=len(ideas),
            surviving=len(surviving_ideas),
            discarded=len(discarded_ideas),
        )

        ctx.report_output({
            "full_prompt": prompt,
            "status": "success",
            "message": f"{len(ideas)} ideas -> {len(surviving_ideas)} surviving, {len(discarded_ideas)} duplicates",
            "surviving_count": len(surviving_ideas),
            "discarded_count": len(discarded_ideas),
            "surviving_ideas": [
                {
                    "title": i.get("title", ""),
                    "summary": i.get("summary", ""),
                    "post_type": i.get("post_type", "free"),
                    "similarity_score": i.get("similarity_score", 0),
                    "reason": i.get("dedup_reason", ""),
                }
                for i in surviving_ideas
            ],
            "discarded_ideas": [
                {
                    "title": d.title,
                    "similarity_score": d.similarity_score,
                    "similar_to": d.similar_to,
                    "reason": d.reason,
                }
                for d in discarded_ideas
            ],
        })

        return DedupIdeasLLMOutput(
            surviving_ideas=surviving_ideas,
            discarded_ideas=discarded_ideas,
            surviving_count=len(surviving_ideas),
            discarded_count=len(discarded_ideas),
            status="success",
        )

    except Exception as e:
        logger.error("dedup_ideas_llm_failed", error=str(e))
        ctx.report_output({
            "status": "error",
            "error": str(e),
        })
        raise RuntimeError(f"Failed to dedup ideas: {e}") from e


async def draft_article(
    ctx,
    params: DraftArticleInput,
) -> DraftArticleOutput:
    """
    Generate a full article draft.

    The article_prompt (from ctx.get_config) controls all writing style, constraints,
    and formatting rules. This function just assembles the inputs and calls the LLM.
    """
    config = _get_llm_config(ctx, params.llm_config)
    title = params.title
    source_content = params.source_content
    constraints = params.constraints
    idea_summary = params.idea_summary
    idea_inspiration = params.idea_inspiration
    post_type = params.post_type

    # Get article prompt from config (required)
    article_prompt_config = ctx.get_config("article_prompt")
    if article_prompt_config:
        article_prompt = article_prompt_config.get("text", article_prompt_config) if isinstance(article_prompt_config, dict) else str(article_prompt_config)
    else:
        article_prompt = params.article_prompt or "Write in a conversational, engaging tone."

    word_count_target = constraints.word_count if constraints else 1500

    ctx.report_input({
        "title": title,
        "idea_summary": idea_summary,
        "idea_inspiration": idea_inspiration,
        "post_type": post_type,
        "word_count_target": word_count_target,
        "has_source_content": bool(source_content),
        "article_prompt_source": "config" if article_prompt_config else "param/default",
        "article_prompt": article_prompt[:200] + "..." if len(article_prompt) > 200 else article_prompt,
        "provider": config["provider"],
        "model": config["model"],
    })

    # Build source context (if source-derived topic)
    source_section = ""
    if source_content:
        source_section = f"""
<source_article>
{source_content[:8000]}
</source_article>

Write an ORIGINAL article inspired by this source. Do NOT copy - transform and add value.
"""

    # Build idea context
    idea_section = ""
    if idea_summary or idea_inspiration or post_type:
        idea_section = "\n<input_context>\n"
        if idea_summary:
            idea_section += f"IDEA SUMMARY: {idea_summary}\n"
        if idea_inspiration:
            idea_section += f"INSPIRATION: {idea_inspiration}\n"
        if post_type:
            content_type = "Premium/Paid" if post_type == "paid" else "Free"
            idea_section += f"CONTENT TYPE: {content_type}\n"
        idea_section += "</input_context>\n"

    prompt = f"""{article_prompt}

TOPIC: {title}
TARGET WORD COUNT: {word_count_target} words
{idea_section}{source_section}
Write the complete article in Markdown format:"""

    try:
        # Use title as cache key for draft_article
        cache_key = f"draft_article:{title[:100]}" if title else None
        draft = await _call_llm(prompt, config, max_tokens=4000, cache_key=cache_key)

        # Count words (rough)
        word_count = len(draft.split())

        logger.info("article_drafted", title=title[:50], words=word_count)

        ctx.report_output({
            "full_prompt": prompt,
            "draft_content": draft,
            "word_count": word_count,
            "status": "success",
        })

        return DraftArticleOutput(
            draft_content=draft,
            word_count=word_count,
            status="success",
            article_prompt=article_prompt,
        )

    except Exception as e:
        logger.error("article_draft_failed", error=str(e))
        ctx.report_output({
            "status": "error",
            "error": str(e),
        })
        return DraftArticleOutput(status="error")


async def generate_image_prompt(
    ctx,
    params: GenerateImagePromptInput,
) -> GenerateImagePromptOutput:
    """
    Generate an image prompt from article content.

    Takes user's image_prompt (style instructions) and combines with article content
    to create a detailed prompt for image generation.
    """
    config = _get_llm_config(ctx, params.llm_config)
    article_content = params.article_content
    title = params.title
    image_prompt = params.image_prompt or "Minimalist, modern aesthetic. Clean composition with subtle gradients."

    ctx.report_input({
        "title": title,
        "article_content_length": len(article_content) if article_content else 0,
        "image_prompt": image_prompt,
        "provider": config["provider"],
        "model": config["model"],
    })

    prompt = f"""Based on this article, create a compelling image prompt for an AI image generator.

ARTICLE TOPIC: {title}

ARTICLE EXCERPT:
{article_content[:2000]}

STYLE INSTRUCTIONS:
{image_prompt}

Create an image prompt that:
1. Captures the essence of the article
2. Would work as a header/hero image
3. Is abstract/conceptual (no text, no specific people)
4. Follows the style instructions above

Respond with ONLY the image prompt, no explanation. Max 200 words."""

    try:
        final_image_prompt = await _call_llm(prompt, config, max_tokens=300)

        logger.info("image_prompt_generated", title=title[:50])

        ctx.report_output({
            "full_prompt": prompt,
            "final_image_prompt": final_image_prompt.strip(),
            "status": "success",
        })

        return GenerateImagePromptOutput(
            final_image_prompt=final_image_prompt.strip(),
            status="success",
        )

    except Exception as e:
        logger.error("image_prompt_failed", error=str(e))
        ctx.report_output({
            "status": "error",
            "error": str(e),
        })
        return GenerateImagePromptOutput(final_image_prompt="", status="error")


def _get_image_config(ctx) -> dict:
    """Get image generation configuration from context secrets (team .env)."""
    return {
        "provider": ctx.get_secret("IMAGE_PROVIDER") or "placeholder",  # placeholder, openai, stable-diffusion, gemini
        "image_model": ctx.get_secret("IMAGE_MODEL") or "gemini-3-pro-image-preview",
        "openai_api_key": ctx.get_secret("OPENAI_API_KEY"),
        "google_api_key": ctx.get_secret("GOOGLE_API_KEY"),
        "sd_api_url": ctx.get_secret("SD_API_URL") or "http://localhost:7860",  # Automatic1111/ComfyUI
    }


async def _generate_image_with_config(
    ctx,
    image_prompt: str,
    config: dict,
) -> GenerateImageOutput:
    """
    Internal: Generate image with explicit config dict.
    """
    provider = config["provider"]

    ctx.report_input({
        "image_prompt": image_prompt[:200] + "..." if len(image_prompt) > 200 else image_prompt,
        "provider": provider,
        "model": config.get("image_model"),
    })

    if provider == "placeholder":
        # Development mode - return a placeholder image
        import hashlib
        prompt_hash = hashlib.md5(image_prompt.encode()).hexdigest()[:6]
        placeholder_url = f"https://placehold.co/1792x1024/1a1a2e/eaeaea?text=AI+Image+{prompt_hash}"

        logger.info("placeholder_image_generated", prompt=image_prompt[:50])

        ctx.report_output({
            "image_url": placeholder_url,
            "provider": "placeholder",
            "status": "success",
        })

        return GenerateImageOutput(
            image_url=placeholder_url,
            status="success",
        )

    elif provider == "stable-diffusion":
        sd_api_url = config["sd_api_url"]

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{sd_api_url}/sdapi/v1/txt2img",
                    json={
                        "prompt": image_prompt,
                        "negative_prompt": "text, watermark, signature, blurry, low quality",
                        "width": 1792,
                        "height": 1024,
                        "steps": 20,
                        "cfg_scale": 7,
                    },
                )
                response.raise_for_status()
                data = response.json()

                image_b64 = data["images"][0]
                image_url = f"data:image/png;base64,{image_b64}"

                logger.info("sd_image_generated", prompt=image_prompt[:50])

                ctx.report_output({
                    "image_url": image_url[:100] + "...",
                    "provider": "stable-diffusion",
                    "status": "success",
                })

                return GenerateImageOutput(
                    image_url=image_url,
                    status="success",
                )

        except Exception as e:
            logger.error("sd_image_generation_failed", error=str(e))
            ctx.report_output({
                "status": "error",
                "error": str(e),
                "provider": "stable-diffusion",
            })
            return GenerateImageOutput(status="error")

    elif provider == "openai":
        openai_api_key = config["openai_api_key"]

        if not openai_api_key:
            ctx.report_output({
                "status": "skipped",
                "reason": "no_api_key",
            })
            return GenerateImageOutput(status="skipped", reason="no_api_key")

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={
                        "Authorization": f"Bearer {openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "dall-e-3",
                        "prompt": image_prompt,
                        "n": 1,
                        "size": "1792x1024",
                        "quality": "standard",
                    },
                )
                response.raise_for_status()
                data = response.json()

                image_url = data["data"][0]["url"]

                logger.info("dalle_image_generated", prompt=image_prompt[:50])

                ctx.report_output({
                    "image_url": image_url,
                    "provider": "openai",
                    "status": "success",
                })

                return GenerateImageOutput(
                    image_url=image_url,
                    status="success",
                )

        except Exception as e:
            logger.error("dalle_image_generation_failed", error=str(e))
            ctx.report_output({
                "status": "error",
                "error": str(e),
                "provider": "openai",
            })
            return GenerateImageOutput(status="error")

    elif provider == "gemini":
        google_api_key = config["google_api_key"]
        image_model = config.get("image_model", "gemini-3-pro-image-preview")

        if not google_api_key:
            ctx.report_output({
                "status": "skipped",
                "reason": "no_api_key",
            })
            return GenerateImageOutput(status="skipped", reason="no_api_key")

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{image_model}:generateContent"

            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    url,
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": google_api_key,
                    },
                    json={
                        "contents": [
                            {
                                "parts": [{"text": f"Generate an image: {image_prompt}"}]
                            }
                        ],
                        "generationConfig": {
                            "responseModalities": ["TEXT", "IMAGE"],
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()

                try:
                    image_data = data["candidates"][0]["content"]["parts"][0]["inlineData"]
                    mime_type = image_data.get("mimeType", "image/png")
                    image_b64 = image_data["data"]
                    image_url = f"data:{mime_type};base64,{image_b64}"

                    logger.info("gemini_image_generated", prompt=image_prompt[:50])

                    ctx.report_output({
                        "image_url": image_url[:100] + "...",
                        "provider": "gemini",
                        "model": image_model,
                        "status": "success",
                    })

                    return GenerateImageOutput(
                        image_url=image_url,
                        status="success",
                    )
                except (KeyError, IndexError) as e:
                    logger.error("gemini_image_parse_error", error=str(e), full_response=data)
                    ctx.report_output({
                        "status": "error",
                        "error": f"Failed to parse Gemini image response: {e}",
                        "provider": "gemini",
                    })
                    return GenerateImageOutput(status="error")

        except Exception as e:
            logger.error("gemini_image_generation_failed", error=str(e))
            ctx.report_output({
                "status": "error",
                "error": str(e),
                "provider": "gemini",
            })
            return GenerateImageOutput(status="error")

    else:
        ctx.report_output({
            "status": "error",
            "error": f"Unknown image provider: {provider}",
        })
        return GenerateImageOutput(status="error")


async def generate_image(
    ctx,
    params: GenerateImageInput,
) -> GenerateImageOutput:
    """
    Generate image using configured provider.

    Supported providers:
    - openai: DALL-E 3 (requires OPENAI_API_KEY)
    - stable-diffusion: Local Stable Diffusion API (Automatic1111/ComfyUI)
    - gemini: Gemini 3 Pro Image (Nano Banana Pro)
    - placeholder: Returns a placeholder image URL (for development)
    """
    config = _get_image_config(ctx)
    image_prompt = params.image_prompt

    # Apply config override if provided
    if params.image_config:
        if params.image_config.provider:
            config["provider"] = params.image_config.provider
        if params.image_config.model:
            config["image_model"] = params.image_config.model

    return await _generate_image_with_config(ctx, image_prompt, config)


async def embed_image_prompts(
    ctx,
    params: EmbedImagePromptsInput,
) -> EmbedImagePromptsOutput:
    """
    Embed image prompts as placeholders in the article (prompts_only mode).

    Instead of generating actual images, this embeds the full prompt text
    in a formatted block so users can:
    - Review and refine prompts before generating
    - Generate images manually with their preferred tool
    - Control costs by only generating images they actually want
    """
    draft_content = params.draft_content
    hero_prompt = params.hero_image_prompt
    inline_prompts = params.inline_image_prompts or []
    image_model = params.image_model

    # Get image styles from config
    hero_style_config = ctx.get_config("hero_image_style")
    hero_image_style = ""
    if hero_style_config:
        hero_image_style = hero_style_config.get("text", hero_style_config) if isinstance(hero_style_config, dict) else str(hero_style_config)
    else:
        hero_image_style = params.hero_image_style or ""

    inline_style_config = ctx.get_config("inline_image_style")
    inline_image_style = ""
    if inline_style_config:
        inline_image_style = inline_style_config.get("text", inline_style_config) if isinstance(inline_style_config, dict) else str(inline_style_config)
    else:
        inline_image_style = params.inline_image_style or ""

    ctx.report_input({
        "draft_length": len(draft_content),
        "has_hero_prompt": bool(hero_prompt),
        "inline_prompts_count": len(inline_prompts),
        "hero_style_source": "config" if hero_style_config else "param/default",
        "inline_style_source": "config" if inline_style_config else "param/default",
        "image_model": image_model,
    })

    # Build full hero prompt with style prefix
    full_hero_prompt = hero_prompt
    if hero_image_style and hero_prompt:
        full_hero_prompt = f"{hero_image_style}\n\n{hero_prompt}"

    # Format the hero image prompt block using markdown blockquote
    # Prefix each line with "> " to keep it in the blockquote
    hero_block = ""
    if full_hero_prompt:
        hero_lines = full_hero_prompt.split('\n')
        hero_quoted = '\n'.join(f"> {line}" if line.strip() else ">" for line in hero_lines)
        hero_block = f"""
---

> **🖼️ HERO IMAGE PROMPT**
>
{hero_quoted}

---

"""

    # Prepend hero image block to the article
    modified_content = hero_block + draft_content

    # Insert inline image prompts after their designated sections
    for img in inline_prompts:
        after_section = img.get("after_section", "")
        prompt = img.get("prompt", "")

        if not after_section or not prompt:
            continue

        # Build full inline prompt with style prefix
        full_inline_prompt = prompt
        if inline_image_style:
            full_inline_prompt = f"{inline_image_style}\n\n{prompt}"

        # Create the inline image placeholder block using markdown blockquote
        # Prefix each line with "> " to keep it in the blockquote
        inline_lines = full_inline_prompt.split('\n')
        inline_quoted = '\n'.join(f"> {line}" if line.strip() else ">" for line in inline_lines)
        inline_block = f"""

> **🖼️ INLINE IMAGE PROMPT**
>
{inline_quoted}

"""

        # Find the section heading and insert after it (and its content paragraph)
        # Look for markdown heading patterns
        import re
        patterns = [
            rf"(#{{1,6}}\s*{re.escape(after_section)}.*?\n)",  # Markdown heading (escaped braces for f-string)
            rf"(\*\*{re.escape(after_section)}\*\*.*?\n)",     # Bold heading
        ]

        inserted = False
        for pattern in patterns:
            match = re.search(pattern, modified_content, re.IGNORECASE)
            if match:
                insert_pos = match.end()
                modified_content = modified_content[:insert_pos] + inline_block + modified_content[insert_pos:]
                inserted = True
                break

        if not inserted:
            logger.warning("inline_image_section_not_found", section=after_section)

    ctx.report_output({
        "draft_content": modified_content[:500] + "..." if len(modified_content) > 500 else modified_content,
        "status": "success",
    })

    return EmbedImagePromptsOutput(
        draft_content=modified_content,
        status="success",
    )


async def inject_placeholders(
    ctx,
    params: InjectPlaceholdersInput,
) -> InjectPlaceholdersOutput:
    """
    Inject image and paywall placeholders into article and compile final image prompts.

    This node:
    1. Takes the raw article and image analysis (which knows WHERE images should go)
    2. Injects simple <image:1>, <image:2> placeholders at those locations
    3. Injects <paywall> placeholder for paid content (at ~35% mark)
    4. Compiles final image prompts by combining base_style + description
    5. Returns the article with placeholders AND the compiled prompts as separate deliverables

    The output is a "production package" ready for manual editing in Substack.
    """
    import re

    draft_content = params.draft_content
    image_analysis = params.image_analysis
    post_type = params.post_type
    hero_image_style = params.hero_image_style or ""
    inline_image_style = params.inline_image_style or ""

    # Get styles from config if available (override params)
    hero_style_config = ctx.get_config("hero_image_style")
    if hero_style_config:
        hero_image_style = hero_style_config.get("text", hero_style_config) if isinstance(hero_style_config, dict) else str(hero_style_config)

    inline_style_config = ctx.get_config("inline_image_style")
    if inline_style_config:
        inline_image_style = inline_style_config.get("text", inline_style_config) if isinstance(inline_style_config, dict) else str(inline_style_config)

    ctx.report_input({
        "draft_length": len(draft_content),
        "has_image_analysis": image_analysis is not None,
        "post_type": post_type,
        "has_hero_style": bool(hero_image_style),
        "has_inline_style": bool(inline_image_style),
    })

    modified_content = draft_content
    image_prompts: List[ImagePromptOutput] = []
    image_counter = 1

    # --- HERO IMAGE (<image:1>) ---
    # Insert after the first paragraph (after intro hook)
    if image_analysis and image_analysis.hero_image:
        hero = image_analysis.hero_image
        placeholder = f"<image:{image_counter}>"

        # Compile final prompt
        final_prompt = hero.prompt
        if hero_image_style:
            final_prompt = f"{hero_image_style}\n\n{hero.prompt}"

        image_prompts.append(ImagePromptOutput(
            placeholder=placeholder,
            placement="hero",
            description_prompt=hero.prompt,
            final_prompt=final_prompt,
            alt_text=hero.alt_text,
        ))

        # Find first paragraph break and insert placeholder
        # Look for double newline after first substantial paragraph
        first_para_match = re.search(r'(^#.*?\n\n.+?\n)\n', modified_content, re.MULTILINE | re.DOTALL)
        if first_para_match:
            insert_pos = first_para_match.end() - 1  # Before the second newline
            modified_content = modified_content[:insert_pos] + f"\n\n{placeholder}\n" + modified_content[insert_pos:]
        else:
            # Fallback: insert after first double newline
            first_break = modified_content.find('\n\n')
            if first_break > 0:
                insert_pos = first_break + 2
                modified_content = modified_content[:insert_pos] + f"{placeholder}\n\n" + modified_content[insert_pos:]

        image_counter += 1

    # --- SUPPLEMENTARY IMAGES (<image:2>, etc.) ---
    if image_analysis and image_analysis.supplementary_images:
        for img in image_analysis.supplementary_images:
            placeholder = f"<image:{image_counter}>"

            # Compile final prompt
            final_prompt = img.prompt
            if inline_image_style:
                final_prompt = f"{inline_image_style}\n\n{img.prompt}"

            image_prompts.append(ImagePromptOutput(
                placeholder=placeholder,
                placement=f"after: {img.after_section}",
                description_prompt=img.prompt,
                final_prompt=final_prompt,
                alt_text=img.alt_text,
            ))

            # Find the section heading and insert placeholder after it
            patterns = [
                rf"(#{{1,6}}\s*{re.escape(img.after_section)}.*?\n)",  # Markdown heading
                rf"(\*\*{re.escape(img.after_section)}\*\*.*?\n)",     # Bold heading
            ]

            inserted = False
            for pattern in patterns:
                match = re.search(pattern, modified_content, re.IGNORECASE)
                if match:
                    insert_pos = match.end()
                    modified_content = modified_content[:insert_pos] + f"\n{placeholder}\n" + modified_content[insert_pos:]
                    inserted = True
                    break

            if not inserted:
                logger.warning("inline_image_section_not_found", section=img.after_section, placeholder=placeholder)

            image_counter += 1

    # --- PAYWALL (<paywall>) ---
    has_paywall = False
    if post_type == "paid":
        # Insert paywall at approximately 35% into the content
        # Find a good break point (paragraph boundary) near that mark
        content_length = len(modified_content)
        target_pos = int(content_length * 0.35)

        # Find the nearest paragraph break (double newline) after target position
        search_start = max(0, target_pos - 200)
        search_region = modified_content[search_start:target_pos + 500]

        # Find paragraph breaks in the search region
        para_breaks = [m.start() + search_start for m in re.finditer(r'\n\n', search_region)]

        if para_breaks:
            # Find the break closest to target_pos
            best_break = min(para_breaks, key=lambda x: abs(x - target_pos))
            insert_pos = best_break + 2  # After the double newline
            modified_content = modified_content[:insert_pos] + "<paywall>\n\n" + modified_content[insert_pos:]
            has_paywall = True
        else:
            # Fallback: insert at target position
            modified_content = modified_content[:target_pos] + "\n\n<paywall>\n\n" + modified_content[target_pos:]
            has_paywall = True

    ctx.report_output({
        "draft_preview": modified_content[:500] + "..." if len(modified_content) > 500 else modified_content,
        "image_prompts_count": len(image_prompts),
        "image_prompts": [{"placeholder": p.placeholder, "placement": p.placement} for p in image_prompts],
        "has_paywall": has_paywall,
        "status": "success",
    })

    return InjectPlaceholdersOutput(
        draft_content=modified_content,
        image_prompts=image_prompts,
        has_paywall=has_paywall,
        status="success",
    )


async def inject_images(
    ctx,
    params: InjectImagesInput,
) -> InjectImagesOutput:
    """
    Unified image injection node (v5.3+).

    This node uses the LLM to:
    1. Analyze the article and identify optimal image placement
    2. INSERT <image:1>, <image:2>, etc. placeholders directly into the article
    3. Optionally INSERT <paywall> for paid content
    4. Generate image prompts for each placeholder
    5. Compile final prompts by combining base_style + description

    Returns:
    - draft_content: Article with placeholders already inserted
    - image_prompts: Dict keyed by placeholder name (e.g., "image:1")
    """
    config = _get_llm_config(ctx, params.llm_config)
    draft_content = params.draft_content
    post_type = params.post_type
    min_images = params.min_images
    max_images = params.max_images

    # Get image styles from config
    hero_style_config = ctx.get_config("hero_image_style")
    hero_image_style = ""
    if hero_style_config:
        hero_image_style = hero_style_config.get("text", hero_style_config) if isinstance(hero_style_config, dict) else str(hero_style_config)

    inline_style_config = ctx.get_config("inline_image_style")
    inline_image_style = ""
    if inline_style_config:
        inline_image_style = inline_style_config.get("text", inline_style_config) if isinstance(inline_style_config, dict) else str(inline_style_config)

    # Build paywall instruction based on post_type
    if post_type == "paid":
        paywall_instruction = """### PAYWALL INSTRUCTION
This is PAID content. You MUST also insert a `<paywall>` placeholder:
- Place `<paywall>` approximately 30-40% into the article
- Content BEFORE <paywall> is the free preview (hook the reader, demonstrate value)
- Content AFTER <paywall> is premium content (deep insights, tactics, analysis)
"""
    else:
        paywall_instruction = """### PAYWALL INSTRUCTION
This is FREE content. Do NOT insert any <paywall> placeholder.
"""

    # Build response schema
    response_schema = f"Respond with JSON matching this schema:\n```json\n{json.dumps(LLMInjectImagesResponse.model_json_schema(), indent=2)}\n```"

    prompt = INJECT_IMAGES_PROMPT.format(
        draft_content=draft_content[:12000],  # Limit input size
        min_images=min_images,
        max_images=max_images,
        paywall_instruction=paywall_instruction,
        response_schema=response_schema,
    )

    ctx.report_input({
        "draft_length": len(draft_content),
        "post_type": post_type,
        "min_images": min_images,
        "max_images": max_images,
        "has_hero_style": bool(hero_image_style),
        "has_inline_style": bool(inline_image_style),
        "provider": config["provider"],
        "model": config["model"],
    })

    try:
        validated_response = await call_llm_validated(
            prompt=prompt,
            config=config,
            response_model=LLMInjectImagesResponse,
            max_tokens=7000,  # Room for article + prompts
            max_retries=1,  # Limit retries for speed
        )

        modified_content = validated_response.modified_article
        raw_prompts = validated_response.image_prompts

        # Check if paywall was inserted
        has_paywall = "<paywall>" in modified_content.lower()

        # Compile final prompts by adding base style
        compiled_prompts: Dict[str, CompiledImagePrompt] = {}
        for key, prompt_item in raw_prompts.items():
            # Determine which style to use (hero for image:1, inline for others)
            is_hero = key == "image:1"
            base_style = hero_image_style if is_hero else inline_image_style

            # Combine style + description
            if base_style:
                final_prompt = f"{base_style}\n\n{prompt_item.prompt}"
            else:
                final_prompt = prompt_item.prompt

            compiled_prompts[key] = CompiledImagePrompt(
                placeholder=key,
                description_prompt=prompt_item.prompt,
                final_prompt=final_prompt,
                alt_text=prompt_item.alt_text,
            )

        logger.info(
            "images_injected",
            image_count=len(compiled_prompts),
            has_paywall=has_paywall,
        )

        ctx.report_output({
            "draft_preview": modified_content[:500] + "..." if len(modified_content) > 500 else modified_content,
            "image_prompts": {k: {"placeholder": v.placeholder, "alt_text": v.alt_text} for k, v in compiled_prompts.items()},
            "has_paywall": has_paywall,
            "status": "success",
        })

        return InjectImagesOutput(
            draft_content=modified_content,
            image_prompts=compiled_prompts,
            has_paywall=has_paywall,
            status="success",
        )

    except Exception as e:
        logger.error("inject_images_failed", error=str(e))
        ctx.report_output({
            "status": "error",
            "error": str(e),
        })
        return InjectImagesOutput(status="error")


async def generate_images_from_prompts(
    ctx,
    params: GenerateImagesFromPromptsInput,
) -> GenerateImagesFromPromptsOutput:
    """
    Generate actual images from compiled image prompts.

    Takes the image_prompts dict from inject_images and generates
    images using the configured image provider.

    If generate_images=False, skips generation and returns status="skipped".

    Returns:
    - generated_images: Dict keyed by placeholder with image URLs
    """
    # Check if we should skip image generation
    if not params.generate_images:
        logger.info("generate_images_skipped", reason="generate_images=false")
        ctx.report_input({"generate_images": False})
        ctx.report_output({"status": "skipped", "reason": "generate_images=false"})
        return GenerateImagesFromPromptsOutput(status="skipped")

    image_config = _get_image_config(ctx)

    # Log raw param value for debugging
    logger.info(
        "generate_images_from_prompts_input",
        image_model_param=params.image_model,
        image_model_type=type(params.image_model).__name__,
        generate_images=params.generate_images,
        prompt_count=len(params.image_prompts),
    )

    # Apply model override if provided
    if params.image_model:
        # Map friendly names to provider/model
        model_map = {
            "DALL-E 3": ("openai", "dall-e-3"),
            "Nano Banana Pro": ("gemini", "gemini-3-pro-image-preview"),
            "Stable Diffusion (Local)": ("stable-diffusion", None),
            "Placeholder (Dev)": ("placeholder", None),
        }
        if params.image_model in model_map:
            provider, model = model_map[params.image_model]
            image_config["provider"] = provider
            if model:
                image_config["image_model"] = model
            logger.info(
                "image_model_resolved",
                input_model=params.image_model,
                resolved_provider=provider,
                resolved_model=model,
            )
        else:
            logger.warning(
                "image_model_not_in_map",
                input_model=params.image_model,
                available_models=list(model_map.keys()),
            )

    # Apply config override if provided
    if params.image_config:
        if params.image_config.provider:
            image_config["provider"] = params.image_config.provider
        if params.image_config.model:
            image_config["image_model"] = params.image_config.model

    image_prompts = params.image_prompts
    provider = image_config["provider"]

    ctx.report_input({
        "prompt_count": len(image_prompts),
        "placeholders": list(image_prompts.keys()),
        "provider": provider,
        "model": image_config.get("image_model"),
    })

    generated_images: Dict[str, GeneratedImageResult] = {}
    total_failed = 0

    for placeholder, prompt_data in image_prompts.items():
        try:
            # Use the final_prompt which has base_style already combined
            final_prompt = prompt_data.final_prompt or prompt_data.description_prompt

            result = await _generate_image_with_config(
                ctx,
                image_prompt=final_prompt,
                config=image_config,
            )

            if result.status == "success" and result.image_url:
                generated_images[placeholder] = GeneratedImageResult(
                    placeholder=placeholder,
                    image_url=result.image_url,
                    prompt_used=final_prompt,
                )
                logger.info(
                    "image_generated",
                    placeholder=placeholder,
                    provider=provider,
                )
            else:
                total_failed += 1
                logger.warning(
                    "image_generation_failed",
                    placeholder=placeholder,
                    reason=result.reason,
                )
        except Exception as e:
            total_failed += 1
            logger.error(
                "image_generation_error",
                placeholder=placeholder,
                error=str(e),
            )

    ctx.report_output({
        "total_generated": len(generated_images),
        "total_failed": total_failed,
        "placeholders_generated": list(generated_images.keys()),
        "status": "success" if generated_images else "partial" if total_failed else "error",
    })

    return GenerateImagesFromPromptsOutput(
        generated_images=generated_images,
        total_generated=len(generated_images),
        total_failed=total_failed,
        status="success" if generated_images else "error",
    )


async def refine_draft(
    ctx,
    params: RefineDraftInput,
) -> RefineDraftOutput:
    """
    Refine a draft based on feedback.
    """
    config = _get_llm_config(ctx, params.llm_config)
    draft_content = params.draft_content
    feedback = params.feedback

    ctx.report_input({
        "draft_content": draft_content,
        "feedback": feedback,
        "provider": config["provider"],
        "model": config["model"],
    })

    prompt = f"""Refine the following article based on the feedback provided.

CURRENT DRAFT:
{draft_content}

FEEDBACK:
{feedback}

Provide the refined article in Markdown format. Make targeted improvements based on the feedback while preserving the overall structure and voice."""

    try:
        refined = await _call_llm(prompt, config, max_tokens=4000)
        word_count = len(refined.split())

        logger.info("draft_refined", feedback=feedback[:50])

        ctx.report_output({
            "full_prompt": prompt,
            "refined_content": refined,
            "word_count": word_count,
            "status": "success",
        })

        return RefineDraftOutput(
            refined_content=refined,
            word_count=word_count,
            status="success",
        )

    except Exception as e:
        logger.error("draft_refinement_failed", error=str(e))
        ctx.report_output({
            "status": "error",
            "error": str(e),
        })
        return RefineDraftOutput(status="error")


# =============================================================================
# LLM Provider Implementations
# =============================================================================

async def _call_llm(
    prompt: str,
    config: dict,
    max_tokens: int = 2000,
    json_mode: bool = False,
    cache_key: Optional[str] = None,
    response_schema: Optional[dict] = None,
    temperature: Optional[float] = None,
) -> str:
    """Call the configured LLM provider with optional caching."""
    provider = config["provider"]

    # Check cache if cache_key provided
    if cache_key:
        cache_data = {"prompt": prompt[:500], "provider": provider, "model": config.get("model"), "key": cache_key}
        cached = _get_cached_response("llm_raw", cache_data)
        if cached:
            logger.info("llm_raw_cache_hit", cache_key=cache_key)
            return cached.get("response", "")

    # Call the actual LLM
    if provider == "openai":
        response = await _call_openai(prompt, config, max_tokens, json_mode, temperature)
    elif provider == "anthropic":
        response = await _call_anthropic(prompt, config, max_tokens, temperature)
    elif provider == "gemini":
        response = await _call_gemini(prompt, config, max_tokens, json_mode, response_schema, temperature)
    elif provider == "ollama":
        response = await _call_ollama(prompt, config, max_tokens, json_mode, temperature)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

    # Save to cache if cache_key provided
    if cache_key:
        _save_to_cache("llm_raw", cache_data, {"response": response})

    return response


async def _call_openai(prompt: str, config: dict, max_tokens: int, json_mode: bool, temperature: Optional[float] = None) -> str:
    """Call OpenAI API."""
    api_key = config["openai_api_key"]
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    async with httpx.AsyncClient(timeout=120) as client:
        request_body = {
            "model": config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }

        if json_mode:
            request_body["response_format"] = {"type": "json_object"}

        # Use explicit temperature if provided, otherwise let OpenAI use default
        if temperature is not None:
            request_body["temperature"] = temperature

        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=request_body,
        )
        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"]


async def _call_anthropic(prompt: str, config: dict, max_tokens: int, temperature: Optional[float] = None) -> str:
    """Call Anthropic API."""
    api_key = config["anthropic_api_key"]
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    request_body = {
        "model": config["model"],
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }

    # Use explicit temperature if provided, otherwise let Anthropic use default
    if temperature is not None:
        request_body["temperature"] = temperature

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            json=request_body,
        )
        response.raise_for_status()
        data = response.json()

        return data["content"][0]["text"]


async def _call_ollama(prompt: str, config: dict, max_tokens: int, json_mode: bool = False, temperature: Optional[float] = None) -> str:
    """Call Ollama API using the chat endpoint."""
    ollama_host = config["ollama_host"]
    model = config["model"]

    logger.info("calling_ollama", host=ollama_host, model=model, prompt_len=len(prompt), json_mode=json_mode, temperature=temperature)

    request_body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "num_predict": max_tokens,
        },
    }

    # Use explicit temperature if provided
    if temperature is not None:
        request_body["options"]["temperature"] = temperature

    # Enable JSON mode if requested
    if json_mode:
        request_body["format"] = "json"

    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(
            f"{ollama_host}/api/chat",
            json=request_body,
        )
        response.raise_for_status()
        data = response.json()

        content = data.get("message", {}).get("content", "")
        logger.info("ollama_response", content_len=len(content), content_preview=content[:200] if content else "EMPTY")

        if not content:
            logger.error("ollama_empty_response", full_response=data)

        return content


def _convert_pydantic_schema_to_gemini(pydantic_schema: dict) -> dict | None:
    """
    Convert a Pydantic JSON schema to Gemini's responseSchema format.

    Gemini expects a simplified schema without:
    - $defs (definitions should be inlined)
    - additionalProperties (Gemini 3 Pro rejects this)
    - title at root level
    - $schema

    Returns None if the schema contains unsupported features that can't be converted.

    See: https://ai.google.dev/gemini-api/docs/structured-output
    """
    has_unsupported_features = False

    def simplify_schema(schema: dict, defs: dict = None) -> dict:
        """Recursively simplify schema for Gemini compatibility."""
        nonlocal has_unsupported_features

        if defs is None:
            defs = schema.get("$defs", {})

        # Handle $ref - inline the referenced definition
        if "$ref" in schema:
            ref_path = schema["$ref"]
            # Extract definition name from "#/$defs/DefinitionName"
            if ref_path.startswith("#/$defs/"):
                def_name = ref_path[8:]
                if def_name in defs:
                    return simplify_schema(defs[def_name], defs)
            return {"type": "object"}  # Fallback

        # Check for additionalProperties - Gemini doesn't support Dict[str, T] schemas
        if "additionalProperties" in schema:
            has_unsupported_features = True
            # Convert to a simple object type (loses type info but won't error)
            return {"type": "object"}

        result = {}

        # Copy allowed fields
        if "type" in schema:
            result["type"] = schema["type"]
        if "description" in schema:
            result["description"] = schema["description"]
        if "enum" in schema:
            result["enum"] = schema["enum"]

        # Handle properties (for objects)
        if "properties" in schema:
            result["properties"] = {
                k: simplify_schema(v, defs)
                for k, v in schema["properties"].items()
            }

        # Handle required fields
        if "required" in schema:
            result["required"] = schema["required"]

        # Handle items (for arrays)
        if "items" in schema:
            result["items"] = simplify_schema(schema["items"], defs)

        return result

    simplified = simplify_schema(pydantic_schema)

    # If the schema has unsupported features, return None so caller can skip schema validation
    if has_unsupported_features:
        return None

    return simplified


async def _call_gemini(
    prompt: str,
    config: dict,
    max_tokens: int,
    json_mode: bool = False,
    response_schema: Optional[dict] = None,
    temperature: Optional[float] = None,
) -> str:
    """
    Call Google Gemini API.

    Supports models: gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash-exp, etc.

    API docs: https://ai.google.dev/gemini-api/docs/text-generation
    """
    api_key = config["google_api_key"]
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")

    model = config["model"]

    # Gemini 2.5 models are "thinking" models that use tokens for internal reasoning.
    # The maxOutputTokens limit includes BOTH thinking tokens AND response tokens.
    # We need to boost the limit to avoid truncation.
    is_thinking_model = "2.5" in model or "thinking" in model.lower()
    effective_max_tokens = max(max_tokens * 4, 8000) if is_thinking_model else max_tokens

    # Use explicit temperature if provided, otherwise use defaults:
    # Lower temperature for structured/scoring tasks (more deterministic)
    # Higher temperature for creative tasks
    if temperature is not None:
        effective_temperature = temperature
    else:
        effective_temperature = 0.2 if json_mode else 0.7

    logger.info(
        "calling_gemini",
        model=model,
        prompt_len=len(prompt),
        json_mode=json_mode,
        has_response_schema=response_schema is not None,
        is_thinking_model=is_thinking_model,
        max_tokens_requested=max_tokens,
        max_tokens_effective=effective_max_tokens,
        temperature=effective_temperature,
    )

    # Gemini uses generateContent endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    request_body = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": effective_max_tokens,
            "temperature": effective_temperature,
        },
    }

    # Enable JSON mode with structured output schema if provided
    if json_mode:
        request_body["generationConfig"]["responseMimeType"] = "application/json"

        # Use response_schema for structured output (Gemini 2.5+)
        # This constrains the model to output valid JSON matching the schema
        if response_schema:
            # Convert Pydantic schema to Gemini format
            # Gemini expects a simpler schema format without $defs, additionalProperties, etc.
            # Returns None if schema contains unsupported features (like Dict[str, T])
            gemini_schema = _convert_pydantic_schema_to_gemini(response_schema)
            if gemini_schema:
                request_body["generationConfig"]["responseSchema"] = gemini_schema
                logger.info("gemini_using_response_schema", schema_keys=list(gemini_schema.get("properties", {}).keys()))
            else:
                # Schema couldn't be converted - rely on JSON mode and prompt instructions only
                logger.info("gemini_skipping_response_schema", reason="schema contains unsupported features")

    # Retry with exponential backoff for rate limits (429)
    max_retries = 3
    for attempt in range(max_retries + 1):
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key,
                },
                json=request_body,
            )

            # Handle rate limiting with retry
            if response.status_code == 429:
                if attempt < max_retries:
                    wait_time = (2 ** attempt) * 2  # 2s, 4s, 8s
                    logger.warning(
                        "gemini_rate_limited_retrying",
                        attempt=attempt + 1,
                        max_retries=max_retries,
                        wait_time=wait_time,
                    )
                    import asyncio
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    response.raise_for_status()  # Raise after all retries exhausted

            response.raise_for_status()
            data = response.json()

            # Extract text from Gemini response structure
            # Response: {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}
            try:
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                logger.info("gemini_response", content_len=len(content), content_preview=content[:200] if content else "EMPTY")
                return content
            except (KeyError, IndexError) as e:
                logger.error("gemini_parse_error", error=str(e), full_response=data)
                raise ValueError(f"Failed to parse Gemini response: {data}")

    # Should never reach here, but just in case
    raise RuntimeError("Gemini call failed after all retries")


# =============================================================================
# TWO-PASS IMAGE DRAFTING
# =============================================================================

async def analyze_article_for_images(
    ctx,
    params: AnalyzeArticleForImagesInput,
) -> AnalyzeArticleForImagesOutput:
    """
    Analyze an article to determine optimal image placement.

    This is pass 1 of the two-pass image drafting approach:
    1. Analyze the article structure and identify where images would add value
    2. Generate prompts for hero image + 0-2 supplementary images
    3. Return structured analysis for batch image generation
    """
    config = _get_llm_config(ctx, params.llm_config)
    draft_content = params.draft_content
    title = params.title
    min_supplementary = params.min_supplementary
    max_supplementary = params.max_supplementary

    if not draft_content:
        ctx.report_input({
            "status": "error",
            "error": "No draft content provided",
        })
        return AnalyzeArticleForImagesOutput(status="error")

    # Generate response schema from Pydantic model
    response_schema = f"Respond with JSON matching this schema:\n```json\n{json.dumps(LLMImageAnalysisResponse.model_json_schema(), indent=2)}\n```"

    prompt = ANALYZE_IMAGES_PROMPT.format(
        draft_content=draft_content[:6000],
        min_supplementary=min_supplementary,
        max_supplementary=max_supplementary,
        response_schema=response_schema,
    )

    ctx.report_input({
        "title": title,
        "min_supplementary": min_supplementary,
        "max_supplementary": max_supplementary,
        "provider": config["provider"],
        "model": config["model"],
        "draft_content_length": len(draft_content),
        "full_prompt": prompt,
    })

    try:
        validated_response = await call_llm_validated(
            prompt=prompt,
            config=config,
            response_model=LLMImageAnalysisResponse,
            max_tokens=3000,  # Increased for elaborate image prompts
        )

        hero_image = HeroImageSpec(
            prompt=validated_response.hero_image.prompt,
            alt_text=validated_response.hero_image.alt_text,
        )

        supplementary_images = [
            ImagePlacement(
                after_section=img.after_section,
                prompt=img.prompt,
                alt_text=img.alt_text,
                rationale=img.rationale,
            )
            for img in validated_response.supplementary_images[:max_supplementary]
        ]

        analysis = ImageAnalysisResult(
            hero_image=hero_image,
            supplementary_images=supplementary_images,
        )

        logger.info(
            "article_analyzed_for_images",
            title=title[:50] if title else "unknown",
            supplementary_count=len(supplementary_images),
        )

        ctx.report_output({
            "hero_prompt": hero_image.prompt,
            "hero_alt_text": hero_image.alt_text,
            "supplementary_images": [
                {"after_section": img.after_section, "prompt": img.prompt, "alt_text": img.alt_text}
                for img in supplementary_images
            ],
            "status": "success",
        })

        return AnalyzeArticleForImagesOutput(
            image_analysis=analysis,
            hero_prompt=hero_image.prompt,
            supplementary_count=len(supplementary_images),
            status="success",
        )

    except Exception as e:
        logger.error("image_analysis_failed", error=str(e))
        ctx.report_output({
            "status": "error",
            "error": str(e),
        })
        return AnalyzeArticleForImagesOutput(status="error")


async def generate_all_images(
    ctx,
    params: GenerateAllImagesInput,
) -> GenerateAllImagesOutput:
    """
    Generate all images for an article in batch.

    Takes the image analysis from analyze_article_for_images and generates
    the hero image + any supplementary images using the configured image provider.

    Supported providers (via IMAGE_PROVIDER env or image_config override):
    - openai: DALL-E 3
    - gemini: Gemini Imagen 3
    - stable-diffusion: Local SD API
    - placeholder: Development placeholder images
    """
    image_config = _get_image_config(ctx)

    # Apply overrides from params if provided
    if params.image_config:
        if params.image_config.provider:
            image_config["provider"] = params.image_config.provider
        if params.image_config.model:
            image_config["image_model"] = params.image_config.model

    image_analysis = params.image_analysis
    hero_image_style = params.hero_image_style
    inline_image_style = params.inline_image_style
    provider = image_config["provider"]

    ctx.report_input({
        "has_hero": bool(image_analysis.hero_image),
        "supplementary_count": len(image_analysis.supplementary_images),
        "has_hero_image_style": bool(hero_image_style),
        "has_inline_image_style": bool(inline_image_style),
        "provider": provider,
        "model": image_config.get("image_model"),
    })

    failed_count = 0

    # Helper to append style to prompt
    def style_hero_prompt(prompt: str) -> str:
        if hero_image_style:
            return f"{prompt}\n\nStyle: {hero_image_style}"
        return prompt

    def style_inline_prompt(prompt: str) -> str:
        if inline_image_style:
            return f"{prompt}\n\nStyle: {inline_image_style}"
        return prompt

    # Helper to generate a single image using configured provider
    async def generate_single(prompt: str) -> str:
        """Generate single image, returns URL or raises exception."""
        result = await generate_image(
            ctx,
            GenerateImageInput(
                image_prompt=prompt,
                image_config=params.image_config,
            ),
        )
        if result.status != "success":
            raise Exception(result.reason or "Image generation failed")
        return result.image_url

    # Generate hero image
    hero_result = None
    if image_analysis.hero_image:
        hero_prompt = style_hero_prompt(image_analysis.hero_image.prompt)
        try:
            url = await generate_single(hero_prompt)
            hero_result = GeneratedImage(
                id="hero",
                url=url,
                alt_text=image_analysis.hero_image.alt_text,
                placement="hero",
                status="success",
            )
            logger.info("hero_image_generated", provider=provider)
        except Exception as e:
            logger.error("hero_image_failed", error=str(e), provider=provider)
            hero_result = GeneratedImage(
                id="hero",
                alt_text=image_analysis.hero_image.alt_text,
                placement="hero",
                status="error",
                error=str(e),
            )
            failed_count += 1

    # Generate supplementary images
    supplementary_results = []
    for i, img_spec in enumerate(image_analysis.supplementary_images):
        img_prompt = style_inline_prompt(img_spec.prompt)
        img_id = f"supplementary_{i}"
        try:
            url = await generate_single(img_prompt)
            supplementary_results.append(GeneratedImage(
                id=img_id,
                url=url,
                alt_text=img_spec.alt_text,
                placement=img_spec.after_section,
                status="success",
            ))
            logger.info("supplementary_image_generated", index=i, after_section=img_spec.after_section, provider=provider)
        except Exception as e:
            logger.error("supplementary_image_failed", index=i, error=str(e), provider=provider)
            supplementary_results.append(GeneratedImage(
                id=img_id,
                alt_text=img_spec.alt_text,
                placement=img_spec.after_section,
                status="error",
                error=str(e),
            ))
            failed_count += 1

    total_generated = (1 if hero_result and hero_result.status == "success" else 0) + \
                      sum(1 for img in supplementary_results if img.status == "success")

    ctx.report_output({
        "provider": provider,
        "model": image_config.get("image_model"),
        "hero_status": hero_result.status if hero_result else "none",
        "supplementary_generated": sum(1 for img in supplementary_results if img.status == "success"),
        "total_generated": total_generated,
        "total_failed": failed_count,
        "status": "success" if total_generated > 0 else "all_failed",
    })

    return GenerateAllImagesOutput(
        hero_image=hero_result,
        supplementary_images=supplementary_results,
        total_generated=total_generated,
        total_failed=failed_count,
        status="success" if total_generated > 0 else "all_failed",
    )


async def stitch_draft(
    ctx,
    params: StitchDraftInput,
) -> StitchDraftOutput:
    """
    Stitch images into the article at the correct positions.

    Takes the original draft and generated images, inserts:
    - Hero image at the very top (before the title if markdown, or as metadata)
    - Supplementary images after their designated section headings
    """
    draft_content = params.draft_content
    hero_image = params.hero_image
    supplementary_images = params.supplementary_images

    ctx.report_input({
        "draft_content_length": len(draft_content) if draft_content else 0,
        "has_hero": bool(hero_image and hero_image.url),
        "supplementary_count": len([img for img in supplementary_images if img.url]),
    })

    if not draft_content:
        ctx.report_output({
            "status": "error",
            "error": "No draft content provided",
        })
        return StitchDraftOutput(status="error")

    final_content = draft_content
    image_count = 0
    hero_url = None

    # Insert supplementary images first (so line numbers don't shift for hero)
    # Process in reverse order to maintain correct positions
    for img in reversed(supplementary_images):
        if not img.url or not img.placement:
            continue

        # Find the section heading and insert image after it
        section_heading = img.placement
        # Try to find the heading in various markdown formats
        patterns = [
            f"## {section_heading}",
            f"### {section_heading}",
            f"# {section_heading}",
        ]

        for pattern in patterns:
            if pattern in final_content:
                # Find end of the heading line
                idx = final_content.find(pattern)
                line_end = final_content.find("\n", idx)
                if line_end == -1:
                    line_end = len(final_content)

                # Insert image markdown after the heading
                image_md = f"\n\n![{img.alt_text}]({img.url})\n"
                final_content = final_content[:line_end] + image_md + final_content[line_end:]
                image_count += 1
                logger.info("supplementary_image_inserted", after_section=section_heading)
                break

    # Insert hero image at the top
    if hero_image and hero_image.url:
        hero_url = hero_image.url
        hero_md = f"![{hero_image.alt_text}]({hero_image.url})\n\n"
        final_content = hero_md + final_content
        image_count += 1
        logger.info("hero_image_inserted")

    ctx.report_output({
        "hero_image_url": hero_url,
        "image_count": image_count,
        "final_content_length": len(final_content),
        "status": "success",
    })

    return StitchDraftOutput(
        final_content=final_content,
        hero_image_url=hero_url,
        image_count=image_count,
        status="success",
    )


async def generate_and_stitch_images(
    ctx,
    params: GenerateAndStitchImagesInput,
) -> GenerateAndStitchImagesOutput:
    """
    Generate images and stitch into draft (if enabled), or pass through with prompts.

    This is a combined node for linear workflow flow:
    - If generate_images=False: returns draft_content as-is (prompts already embedded)
    - If generate_images=True: generates actual images and stitches them into the draft

    Used by the factory workflow v5.0 to simplify conditional logic.
    """
    draft_content = params.draft_content
    image_analysis = params.image_analysis
    generate_images = params.generate_images
    image_config = params.image_config

    # Get image styles from config
    hero_style_config = ctx.get_config("hero_image_style")
    hero_image_style = ""
    if hero_style_config:
        hero_image_style = hero_style_config.get("text", hero_style_config) if isinstance(hero_style_config, dict) else str(hero_style_config)
    else:
        hero_image_style = params.hero_image_style or ""

    inline_style_config = ctx.get_config("inline_image_style")
    inline_image_style = ""
    if inline_style_config:
        inline_image_style = inline_style_config.get("text", inline_style_config) if isinstance(inline_style_config, dict) else str(inline_style_config)
    else:
        inline_image_style = params.inline_image_style or ""

    ctx.report_input({
        "draft_content_length": len(draft_content) if draft_content else 0,
        "generate_images": generate_images,
        "has_hero": bool(image_analysis and image_analysis.hero_image),
        "supplementary_count": len(image_analysis.supplementary_images) if image_analysis else 0,
        "hero_style_source": "config" if hero_style_config else "param/default",
        "inline_style_source": "config" if inline_style_config else "param/default",
    })

    if not draft_content:
        ctx.report_output({
            "status": "error",
            "error": "No draft content provided",
        })
        return GenerateAndStitchImagesOutput(status="error")

    # If not generating images, just pass through the draft with embedded prompts
    if not generate_images:
        logger.info("generate_and_stitch_images_passthrough", reason="generate_images=False")
        ctx.report_output({
            "final_content_length": len(draft_content),
            "images_generated": False,
            "image_count": 0,
            "status": "success",
        })
        return GenerateAndStitchImagesOutput(
            final_content=draft_content,
            hero_image_url=None,
            image_count=0,
            images_generated=False,
            status="success",
        )

    # Generate images is True - generate actual images and stitch them
    logger.info("generate_and_stitch_images_generating",
                hero_style=hero_image_style[:50] if hero_image_style else None,
                inline_style=inline_image_style[:50] if inline_image_style else None)

    # Step 1: Generate all images
    gen_result = await generate_all_images(
        ctx,
        GenerateAllImagesInput(
            image_analysis=image_analysis,
            hero_image_style=hero_image_style,
            inline_image_style=inline_image_style,
            image_config=image_config,
        ),
    )

    if gen_result.status == "all_failed":
        # All image generation failed - return draft with prompts as fallback
        logger.warning("generate_and_stitch_images_all_failed", reason="all image generation failed")
        ctx.report_output({
            "final_content_length": len(draft_content),
            "images_generated": False,
            "image_count": 0,
            "status": "error",
            "error": "All image generation failed",
        })
        return GenerateAndStitchImagesOutput(
            final_content=draft_content,
            hero_image_url=None,
            image_count=0,
            images_generated=False,
            status="error",
        )

    # Step 2: Stitch images into draft
    stitch_result = await stitch_draft(
        ctx,
        StitchDraftInput(
            draft_content=draft_content,
            hero_image=gen_result.hero_image,
            supplementary_images=gen_result.supplementary_images,
        ),
    )

    if stitch_result.status != "success":
        ctx.report_output({
            "status": "error",
            "error": "Stitch draft failed",
        })
        return GenerateAndStitchImagesOutput(status="error")

    logger.info(
        "generate_and_stitch_images_complete",
        image_count=stitch_result.image_count,
        has_hero=bool(stitch_result.hero_image_url),
    )

    ctx.report_output({
        "final_content_length": len(stitch_result.final_content),
        "hero_image_url": stitch_result.hero_image_url,
        "images_generated": True,
        "image_count": stitch_result.image_count,
        "status": "success",
    })

    return GenerateAndStitchImagesOutput(
        final_content=stitch_result.final_content,
        hero_image_url=stitch_result.hero_image_url,
        image_count=stitch_result.image_count,
        images_generated=True,
        status="success",
    )
