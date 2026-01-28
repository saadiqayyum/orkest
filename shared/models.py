"""
Database models for the AI Substack Mirroring Engine.

These models define the domain-specific tables for content mirroring:
- posts: Unified post storage (the "Market" memory)
- target_items: Published content (the "Published" memory)
- production_queue: Active content pipeline state

This is the dg-team's own database schema, separate from the engine's
workflow tracking tables (as per constitution).
"""
import uuid
from datetime import datetime, timezone
from typing import Optional


def utc_now():
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)
from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean,
    DateTime, ForeignKey, Index, Enum, JSON,
    UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship, declarative_base
from pgvector.sqlalchemy import Vector
import enum
import os

Base = declarative_base()

# Embedding dimensions - configurable via environment
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIMENSIONS", "768"))


class QueueStatus(enum.Enum):
    """Status of items in the production queue."""
    PENDING = "pending"
    DRAFTING = "drafting"
    REVIEW = "review"
    PUBLISHED = "published"
    REJECTED = "rejected"


# =============================================================================
# Config → Instance → Artifact Architecture Enums
# =============================================================================

class ConfigType(enum.Enum):
    """Type of configuration stored in configs table."""
    SUBSTACK_AUTH = "SUBSTACK_AUTH"          # Substack session cookies
    API_KEY = "API_KEY"                      # User's own API keys (BYOK)
    PROMPT_TEMPLATE = "PROMPT_TEMPLATE"      # Reusable prompts for LLM
    PUBLICATION_LIST = "PUBLICATION_LIST"   # List of publication handles
    GENERAL = "GENERAL"                      # Arbitrary key-value


class InstanceStatus(enum.Enum):
    """Status of a workflow instance (campaign)."""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


class ArtifactStatus(enum.Enum):
    """Status of artifacts in the pipeline."""
    IDEA = "idea"
    PENDING = "pending"
    DRAFTING = "drafting"
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class ArtifactType(enum.Enum):
    """Type of artifact produced by a workflow."""
    ARTICLE = "article"
    IMAGE = "image"
    REPORT = "report"
    IDEA = "idea"
    SUMMARY = "summary"
    OTHER = "other"


class Post(Base):
    """
    Unified post storage - posts belong to a publication_handle, NOT a target.

    Key insight: Posts are just posts. They belong to a publication_handle.
    Workflows take publication handles as parameters - the DB doesn't need to track
    which target cares about which posts.
    """
    __tablename__ = "posts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Core identification
    url = Column(Text, unique=True, nullable=False, index=True)
    publication_handle = Column(String(100), nullable=False, index=True)

    # Content
    title = Column(Text, nullable=False)
    subtitle = Column(Text)  # Substack subtitle/description (merged - stores whichever is longer)
    content_raw = Column(Text)  # Full scraped content (may be partial if paywalled)
    author = Column(String(255))

    # Substack metadata
    substack_post_id = Column(String(100))
    slug = Column(String(500))
    post_type = Column(String(50))  # 'newsletter', 'podcast', 'thread'
    audience = Column(String(50))   # 'everyone', 'only_paid', 'founding'
    word_count = Column(Integer)
    tags = Column(ARRAY(Text))  # Post tags (e.g., 'System Design', 'Coding Interviews')
    section_name = Column(String(255))  # Section within publication (e.g., 'Deep Dives')

    # Paywall tracking
    is_paywalled = Column(Boolean, default=False)
    has_full_content = Column(Boolean, default=True)

    # Engagement metrics
    likes_count = Column(Integer, default=0, index=True)
    comments_count = Column(Integer, default=0)
    shares_count = Column(Integer, default=0)
    published_at = Column(DateTime(timezone=True), index=True)

    # Embedding for semantic search
    embedding = Column(Vector(EMBEDDING_DIM))

    # Tracking
    discovery_source = Column(String(50))  # 'content_sync', 'publication_discovery'
    first_seen_at = Column(DateTime(timezone=True), default=utc_now)
    content_fetched_at = Column(DateTime(timezone=True))
    metrics_updated_at = Column(DateTime(timezone=True))
    refetch_attempted_at = Column(DateTime(timezone=True))

    __table_args__ = (
        Index('idx_posts_publication', publication_handle),
        Index('idx_posts_published_at', published_at.desc()),
        Index('idx_posts_likes', likes_count.desc()),
        Index('idx_posts_pub_likes', publication_handle, likes_count.desc()),
        Index('idx_posts_has_full_content', has_full_content, postgresql_where=(has_full_content == False)),
    )


class TargetItem(Base):
    """
    The "Published" Memory - stores our published content.

    This table tracks everything we've published to our target publication.
    Used for Hard Dedup (don't recreate what we've already published).
    """
    __tablename__ = "target_items"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Pipeline scoping - which target publication this was published to
    target_publication = Column(String(100), nullable=False, index=True)

    # Link to source post (if this was derived from a post)
    source_ref_id = Column(UUID(as_uuid=True), ForeignKey("posts.id"), nullable=True, index=True)

    # Our published content
    url = Column(Text, unique=True, nullable=False, index=True)  # Our Substack URL
    title = Column(Text, nullable=False)
    content_raw = Column(Text)  # Our published content

    # Embedding for semantic search (Soft Dedup)
    embedding = Column(Vector(EMBEDDING_DIM))

    # Our metrics (tracking our own performance)
    metrics_json = Column(JSONB, default=dict)
    likes_count = Column(Integer, default=0)
    comments_count = Column(Integer, default=0)

    # Timestamps
    published_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    __table_args__ = (
        Index('idx_target_items_published', published_at.desc()),
    )


class ProductionQueueItem(Base):
    """
    The Active State - production queue for content pipeline.

    This is the central locking mechanism to prevent duplication.
    Items move through: pending -> drafting -> review -> published

    CRITICAL: The embedding here is used for Soft Dedup - before adding
    new items, we vector search against this table to prevent duplicates.
    """
    __tablename__ = "production_queue"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Pipeline scoping - which target publication this queue item is for
    target_publication = Column(String(100), nullable=False, index=True)

    # Source reference - references posts.id
    source_ref_id = Column(UUID(as_uuid=True), ForeignKey("posts.id"), nullable=True, index=True)

    # Content info
    title = Column(Text, nullable=False)  # Article title
    topic_type = Column(String(50), default="source_derived")  # "source_derived" or "idea_free" or "idea_paid"

    # CRITICAL: Embedding for Soft Dedup
    embedding = Column(Vector(EMBEDDING_DIM))

    # Pipeline status
    status = Column(
        Enum(QueueStatus, name="queue_status", create_type=False, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=QueueStatus.PENDING,
        index=True
    )

    # Draft content (populated during drafting phase)
    draft_content = Column(Text)  # Markdown article
    draft_image_url = Column(Text)  # Generated image URL
    draft_metadata = Column(JSONB, default=dict)  # Additional draft info

    # Scoring (for prioritization)
    priority_score = Column(Float, default=0.0, index=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
    drafted_at = Column(DateTime(timezone=True))
    published_at = Column(DateTime(timezone=True))

    __table_args__ = (
        Index('idx_queue_status_priority', status, priority_score.desc()),
        Index('idx_queue_source_ref', source_ref_id),
    )


class IdeaSourcePost(Base):
    """
    Track which source posts inspired which ideas.

    Used to filter out already-worked source posts in fetch_top_source_posts.
    Future: decay logic to allow posts to become candidates again after N days.
    """
    __tablename__ = "idea_source_posts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    queue_item_id = Column(UUID(as_uuid=True), ForeignKey("production_queue.id", ondelete="CASCADE"), nullable=False)
    source_post_id = Column(UUID(as_uuid=True), ForeignKey("posts.id", ondelete="CASCADE"), nullable=False)
    target_handle = Column(String(100), nullable=False, index=True)  # Denormalized for fast filtering
    created_at = Column(DateTime(timezone=True), default=utc_now)

    __table_args__ = (
        UniqueConstraint('queue_item_id', 'source_post_id', name='uq_idea_source_posts'),
        Index('idx_idea_source_posts_source', source_post_id),
        Index('idx_idea_source_posts_created', created_at),
    )


class LLMRelevanceCache(Base):
    """
    Cache for LLM-based relevance scores.

    Key: (post_url, keywords_normalized, model)
    Value: relevance score for that post

    TTL: 48 hours (different models may give different scores)

    Caching per-post (not per-publication) means:
    - If a publication adds new posts, we still use cache for older posts
    - Only new posts need LLM scoring
    """
    __tablename__ = "llm_relevance_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Cache key (includes model for score consistency)
    post_url = Column(String(2000), nullable=False, index=True)
    keywords_normalized = Column(String(500), nullable=False, index=True)  # lowercase, sorted, semicolon-joined
    model = Column(String(100), nullable=False, default='unknown')  # LLM model used for scoring

    # Cached data
    score = Column(Integer, nullable=False)  # 0-100 relevance score
    publication_handle = Column(String(255), nullable=False, index=True)  # For grouping results

    # TTL tracking
    cached_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    __table_args__ = (
        UniqueConstraint('post_url', 'keywords_normalized', 'model', name='uq_llm_cache_post_keywords_model'),
        Index('idx_llm_cache_key', post_url, keywords_normalized, model),
        Index('idx_llm_cache_cached_at', cached_at),
    )


# =============================================================================
# Config → Instance → Artifact Architecture Models
# =============================================================================

class Config(Base):
    """
    Credentials & Reusable Assets - the "locker" for app-level configs.

    Value schemas by type:
    - SUBSTACK_AUTH: {"cookies": {...}, "paid_handles": [...]}
    - API_KEY: {"provider": "openai", "key": "sk-..."}
    - PROMPT_TEMPLATE: {"content": "Write an article...", "purpose": "article"}
    - PUBLICATION_LIST: {"handles": ["stratechery", "bytebytego"]}
    - GENERAL: Arbitrary JSONB
    """
    __tablename__ = "configs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    type = Column(
        Enum(ConfigType, name="config_type", create_type=False, values_callable=lambda x: [e.value for e in x]),
        nullable=False
    )
    description = Column(Text)
    value = Column(JSONB, nullable=False, default=dict)
    is_secret = Column(Boolean, default=False)  # Hint for UI to mask values
    tags = Column(ARRAY(Text), default=list)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    # Relationships
    instance_bindings = relationship("InstanceConfigBinding", back_populates="config")

    __table_args__ = (
        Index('idx_configs_type', type),
        Index('idx_configs_tags', tags, postgresql_using='gin'),
    )


class WorkflowTemplate(Base):
    """
    Workflow Catalog - maps UI-friendly names to actual workflow paths.
    """
    __tablename__ = "workflow_templates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text)
    workflow_ref = Column(String(255), nullable=False)  # e.g., "substack/content_sync_v3"
    required_config_types = Column(ARRAY(String), default=list)  # Array of ConfigType values
    default_params = Column(JSONB, default=dict)
    category = Column(String(100))  # e.g., "sync", "generation"
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    # Relationships
    instances = relationship("Instance", back_populates="template")

    __table_args__ = (
        Index('idx_workflow_templates_category', category),
        Index('idx_workflow_templates_active', is_active, postgresql_where=(is_active == True)),
    )


class Instance(Base):
    """
    Campaigns/Jobs - the core "mission" entity.

    Binds template + configs + params. Holds mutable state.
    - `params` is user-set at creation (e.g., source_handles)
    - `state` is workflow-controlled (cursors, progress)
    """
    __tablename__ = "instances"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text)
    template_id = Column(UUID(as_uuid=True), ForeignKey("workflow_templates.id"), nullable=False, index=True)
    params = Column(JSONB, default=dict)  # Instance-specific params
    state = Column(JSONB, default=dict)   # Mutable workflow state
    status = Column(
        Enum(InstanceStatus, name="instance_status", create_type=False, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=InstanceStatus.ACTIVE,
        index=True
    )
    last_run_at = Column(DateTime(timezone=True))
    run_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    # Relationships
    template = relationship("WorkflowTemplate", back_populates="instances")
    config_bindings = relationship("InstanceConfigBinding", back_populates="instance", cascade="all, delete-orphan")
    artifacts = relationship("Artifact", back_populates="instance", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_instances_last_run', last_run_at.desc()),
    )


class InstanceConfigBinding(Base):
    """
    Junction table: which configs are bound to which instance.
    """
    __tablename__ = "instance_config_bindings"

    instance_id = Column(UUID(as_uuid=True), ForeignKey("instances.id", ondelete="CASCADE"), primary_key=True)
    config_id = Column(UUID(as_uuid=True), ForeignKey("configs.id", ondelete="CASCADE"), primary_key=True)
    binding_role = Column(String(100))  # Optional: "auth", "prompts", "sources"

    # Relationships
    instance = relationship("Instance", back_populates="config_bindings")
    config = relationship("Config", back_populates="instance_bindings")

    __table_args__ = (
        Index('idx_instance_config_bindings_instance', instance_id),
        Index('idx_instance_config_bindings_config', config_id),
    )


class Artifact(Base):
    """
    Instance Outputs - replaces production_queue + target_items.

    Scoped by instance_id = no collisions between campaigns.
    Dashboard query: SELECT * FROM artifacts WHERE instance_id = ?
    """
    __tablename__ = "artifacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    instance_id = Column(UUID(as_uuid=True), ForeignKey("instances.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(Text, nullable=False)
    type = Column(
        Enum(ArtifactType, name="artifact_type", create_type=False, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=ArtifactType.ARTICLE
    )
    status = Column(
        Enum(ArtifactStatus, name="artifact_status", create_type=False, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=ArtifactStatus.PENDING,
        index=True
    )
    source_ref_id = Column(UUID(as_uuid=True), ForeignKey("posts.id", ondelete="SET NULL"), index=True)
    source_url = Column(Text)  # Denormalized for quick display
    content_raw = Column(Text)
    artifact_metadata = Column("metadata", JSONB, default=dict)  # Type-specific: word_count, model, prompts_used
    hero_image_url = Column(Text)
    embedding = Column(Vector(EMBEDDING_DIM))  # For soft dedup
    priority_score = Column(Float, default=0.0, index=True)
    published_url = Column(Text)
    published_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
    drafted_at = Column(DateTime(timezone=True))

    # Relationships
    instance = relationship("Instance", back_populates="artifacts")

    __table_args__ = (
        Index('idx_artifacts_type', type),
        Index('idx_artifacts_source_ref', source_ref_id),
        Index('idx_artifacts_published_at', published_at.desc(), postgresql_where=(published_at != None)),
        Index('idx_artifacts_instance_status', instance_id, status),
    )


# Alembic-style migration helper
def create_all_tables(engine):
    """Create all tables in the database."""
    Base.metadata.create_all(engine)


def drop_all_tables(engine):
    """Drop all tables (use with caution!)."""
    Base.metadata.drop_all(engine)
