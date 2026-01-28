-- DG-Team Database Schema
-- Run this fresh: psql -d dg_team -f schema.sql
-- To wipe and recreate: dropdb dg_team && createdb dg_team && psql -d dg_team -f schema.sql

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- ENUMS
-- =============================================================================
CREATE TYPE queue_status AS ENUM ('pending', 'drafting', 'review', 'published');

-- Config → Instance → Artifact architecture enums
CREATE TYPE config_type AS ENUM (
    'SUBSTACK_AUTH',      -- Substack session cookies
    'API_KEY',            -- User's own API keys (BYOK) - OpenAI, etc.
    'PROMPT_TEMPLATE',    -- Reusable prompts for LLM
    'PUBLICATION_LIST',   -- List of publication handles
    'GENERAL'             -- Arbitrary key-value
);

CREATE TYPE instance_status AS ENUM ('active', 'paused', 'stopped', 'completed', 'failed');

CREATE TYPE artifact_status AS ENUM (
    'idea', 'pending', 'drafting', 'draft', 'review',
    'approved', 'published', 'rejected', 'archived'
);

CREATE TYPE artifact_type AS ENUM ('article', 'image', 'report', 'idea', 'summary', 'other');

-- =============================================================================
-- HELPER FUNCTIONS (must be defined before triggers that use them)
-- =============================================================================

CREATE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- TARGET CONFIG - Per-target publication settings
-- Stores default prompts that Factory workflow will use unless overridden
-- =============================================================================
CREATE TABLE target_config (
    target_handle VARCHAR(100) PRIMARY KEY,

    -- Default prompts for content generation (used by Factory workflow)
    article_prompt TEXT,            -- Writing style instructions for article generation
    hero_image_prompt TEXT,         -- Style instructions for hero/featured images
    inline_image_prompt TEXT,       -- Style instructions for supplementary inline images

    -- Substack authentication for accessing paid subscriber content
    -- Stores session cookies (substack.sid, substack.lli) as JSON
    -- These persist for months unless user logs out
    substack_cookies JSONB,

    -- Whitelist of publications where we have paid subscriptions
    -- For these, we MUST get full content (workflow fails if cookies expire)
    paid_subscription_handles TEXT[],

    -- Tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TRIGGER update_target_config_updated_at
    BEFORE UPDATE ON target_config
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- POSTS - Unified post storage
-- =============================================================================
-- Posts are global - they belong to a publication_handle, NOT a target.
-- Workflows take publication handles as parameters.
CREATE TABLE posts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Core identification
    url TEXT UNIQUE NOT NULL,
    publication_handle VARCHAR(100) NOT NULL,

    -- Content
    title TEXT NOT NULL,
    subtitle TEXT,              -- Substack subtitle/description (merged - stores whichever is longer)
    content_raw TEXT,
    author VARCHAR(255),

    -- Substack metadata
    substack_post_id VARCHAR(100),
    slug VARCHAR(500),
    post_type VARCHAR(50),      -- 'newsletter', 'podcast', 'thread'
    audience VARCHAR(50),        -- 'everyone', 'only_paid', 'founding'
    word_count INTEGER,
    tags TEXT[],                -- Post tags (e.g., 'System Design', 'Coding Interviews')
    section_name VARCHAR(255),  -- Section within publication (e.g., 'Deep Dives')

    -- Paywall tracking
    is_paywalled BOOLEAN DEFAULT FALSE,
    has_full_content BOOLEAN DEFAULT TRUE,

    -- Engagement metrics
    likes_count INTEGER DEFAULT 0,
    comments_count INTEGER DEFAULT 0,
    shares_count INTEGER DEFAULT 0,
    published_at TIMESTAMP WITH TIME ZONE,

    -- Embedding for semantic search (768 dimensions for nomic-embed-text)
    embedding vector(768),

    -- Tracking
    discovery_source VARCHAR(50),  -- 'content_sync', 'publication_discovery'
    first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    content_fetched_at TIMESTAMP WITH TIME ZONE,
    metrics_updated_at TIMESTAMP WITH TIME ZONE,
    refetch_attempted_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_posts_url ON posts(url);
CREATE INDEX idx_posts_publication ON posts(publication_handle);
CREATE INDEX idx_posts_published_at ON posts(published_at DESC);
CREATE INDEX idx_posts_likes ON posts(likes_count DESC);
CREATE INDEX idx_posts_pub_likes ON posts(publication_handle, likes_count DESC);
CREATE INDEX idx_posts_has_full_content ON posts(has_full_content) WHERE NOT has_full_content;

-- =============================================================================
-- TARGET ITEMS - Our published content
-- =============================================================================
CREATE TABLE target_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Pipeline scoping
    target_publication VARCHAR(100) NOT NULL,

    -- Link to source post (if derived from a post)
    source_ref_id UUID REFERENCES posts(id) ON DELETE SET NULL,

    -- Our published content
    url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    content_raw TEXT,

    -- Embedding for semantic search
    embedding vector(768),

    -- Our metrics
    metrics_json JSONB DEFAULT '{}',
    likes_count INTEGER DEFAULT 0,
    comments_count INTEGER DEFAULT 0,

    -- Timestamps
    published_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_target_items_url ON target_items(url);
CREATE INDEX idx_target_items_target_publication ON target_items(target_publication);
CREATE INDEX idx_target_items_published_at ON target_items(published_at DESC);
CREATE INDEX idx_target_items_source_ref ON target_items(source_ref_id);

-- =============================================================================
-- PRODUCTION QUEUE - Active pipeline state
-- =============================================================================
CREATE TABLE production_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Pipeline scoping
    target_publication VARCHAR(100) NOT NULL,

    -- Source reference (NULL for gap analysis / original ideas)
    source_ref_id UUID REFERENCES posts(id) ON DELETE SET NULL,

    -- Content info
    title TEXT NOT NULL,  -- Article title
    topic_type VARCHAR(50) DEFAULT 'source_derived',

    -- Embedding for soft dedup
    embedding vector(768),

    -- Pipeline status
    status queue_status NOT NULL DEFAULT 'pending',

    -- Draft content
    draft_content TEXT,
    draft_image_url TEXT,
    draft_metadata JSONB DEFAULT '{}',

    -- Scoring
    priority_score FLOAT DEFAULT 0.0,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    drafted_at TIMESTAMP WITH TIME ZONE,
    published_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_queue_target_publication ON production_queue(target_publication);
CREATE INDEX idx_queue_status ON production_queue(status);
CREATE INDEX idx_queue_priority ON production_queue(priority_score DESC);
CREATE INDEX idx_queue_source_ref ON production_queue(source_ref_id);

CREATE TRIGGER update_production_queue_updated_at
    BEFORE UPDATE ON production_queue
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- IDEA SOURCE POSTS - Track which source posts inspired which ideas
-- =============================================================================
-- Used to filter out already-worked source posts in fetch_top_source_posts.
-- Future: decay logic to allow posts to become candidates again after N days.
CREATE TABLE idea_source_posts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    queue_item_id UUID REFERENCES production_queue(id) ON DELETE CASCADE,
    source_post_id UUID REFERENCES posts(id) ON DELETE CASCADE,
    target_handle VARCHAR(100) NOT NULL,  -- Denormalized for fast filtering
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    UNIQUE(queue_item_id, source_post_id)
);

CREATE INDEX idx_idea_source_posts_target ON idea_source_posts(target_handle);
CREATE INDEX idx_idea_source_posts_source ON idea_source_posts(source_post_id);
CREATE INDEX idx_idea_source_posts_created ON idea_source_posts(created_at);

-- =============================================================================
-- HTTP CACHE - Cache expensive fetch calls to avoid rate limiting
-- =============================================================================
CREATE TABLE http_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Cache key (URL or custom key)
    url TEXT NOT NULL,

    -- Response data
    content TEXT,
    status_code INTEGER,
    content_type VARCHAR(255),

    -- Metadata
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ,  -- NULL means no expiry

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(url)
);

CREATE INDEX idx_http_cache_url ON http_cache(url);
CREATE INDEX idx_http_cache_expires_at ON http_cache(expires_at) WHERE expires_at IS NOT NULL;

-- =============================================================================
-- LLM RELEVANCE CACHE - Cache LLM-based relevance scores for posts
-- =============================================================================
CREATE TABLE llm_relevance_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Cache key (includes model for score consistency)
    post_url VARCHAR(2000) NOT NULL,
    keywords_normalized VARCHAR(500) NOT NULL,  -- lowercase, sorted, semicolon-joined
    model VARCHAR(100) NOT NULL DEFAULT 'unknown',  -- LLM model used for scoring

    -- Cached data
    score INTEGER NOT NULL,  -- 0-100 relevance score
    publication_handle VARCHAR(255) NOT NULL,  -- For grouping results

    -- TTL tracking
    cached_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(post_url, keywords_normalized, model)
);

CREATE INDEX idx_llm_cache_key ON llm_relevance_cache(post_url, keywords_normalized, model);
CREATE INDEX idx_llm_cache_cached_at ON llm_relevance_cache(cached_at);

-- =============================================================================
-- CONFIGS - Credentials & Reusable Assets
-- =============================================================================
-- App-level configs only (dg-team domain). Orkest doesn't persist configs.
--
-- Value schemas by type:
-- - SUBSTACK_AUTH: {"cookies": {"substack.sid": "...", "substack.lli": "..."}, "paid_handles": ["stratechery"]}
-- - API_KEY: {"provider": "openai", "key": "sk-..."}
-- - PROMPT_TEMPLATE: {"content": "Write an article...", "purpose": "article"}
-- - PUBLICATION_LIST: {"handles": ["stratechery", "bytebytego"]}
-- - GENERAL: Arbitrary JSONB

CREATE TABLE configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,  -- Unique within dg-team
    type config_type NOT NULL,
    description TEXT,
    value JSONB NOT NULL DEFAULT '{}',
    is_secret BOOLEAN DEFAULT FALSE,    -- Hint for UI to mask values
    tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_configs_type ON configs(type);
CREATE INDEX idx_configs_slug ON configs(slug);
CREATE INDEX idx_configs_tags ON configs USING GIN(tags);

CREATE TRIGGER update_configs_updated_at
    BEFORE UPDATE ON configs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- WORKFLOW TEMPLATES - Workflow Catalog
-- =============================================================================
-- Maps UI-friendly names to actual workflow paths in manifest.yaml.

CREATE TABLE workflow_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    workflow_ref VARCHAR(255) NOT NULL,  -- e.g., "substack/content_sync_v3"
    required_config_types config_type[] DEFAULT '{}',
    default_params JSONB DEFAULT '{}',
    category VARCHAR(100),               -- e.g., "sync", "generation"
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_workflow_templates_slug ON workflow_templates(slug);
CREATE INDEX idx_workflow_templates_category ON workflow_templates(category);
CREATE INDEX idx_workflow_templates_active ON workflow_templates(is_active) WHERE is_active = TRUE;

CREATE TRIGGER update_workflow_templates_updated_at
    BEFORE UPDATE ON workflow_templates
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- INSTANCES - Campaigns/Jobs
-- =============================================================================
-- The core "mission" entity. Binds template + configs + params. Holds mutable state.

CREATE TABLE instances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    template_id UUID NOT NULL REFERENCES workflow_templates(id),
    params JSONB DEFAULT '{}',           -- Instance-specific params (e.g., source_handles)
    state JSONB DEFAULT '{}',            -- Mutable: {"last_sync": "2024-01-01", "cursor": 55}
    status instance_status NOT NULL DEFAULT 'active',
    last_run_at TIMESTAMPTZ,
    run_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_instances_slug ON instances(slug);
CREATE INDEX idx_instances_template ON instances(template_id);
CREATE INDEX idx_instances_status ON instances(status);
CREATE INDEX idx_instances_last_run ON instances(last_run_at DESC);

CREATE TRIGGER update_instances_updated_at
    BEFORE UPDATE ON instances
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- INSTANCE CONFIG BINDINGS - Junction Table
-- =============================================================================
-- Which configs are bound to which instance

CREATE TABLE instance_config_bindings (
    instance_id UUID NOT NULL REFERENCES instances(id) ON DELETE CASCADE,
    config_id UUID NOT NULL REFERENCES configs(id) ON DELETE CASCADE,
    binding_role VARCHAR(100),           -- Optional: "auth", "prompts", "sources"
    PRIMARY KEY (instance_id, config_id)
);

CREATE INDEX idx_instance_config_bindings_instance ON instance_config_bindings(instance_id);
CREATE INDEX idx_instance_config_bindings_config ON instance_config_bindings(config_id);

-- =============================================================================
-- ARTIFACTS - Instance Outputs
-- =============================================================================
-- Replaces production_queue + target_items. Scoped by instance_id = no collisions.

CREATE TABLE artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    instance_id UUID NOT NULL REFERENCES instances(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    type artifact_type NOT NULL DEFAULT 'article',
    status artifact_status NOT NULL DEFAULT 'pending',
    source_ref_id UUID REFERENCES posts(id) ON DELETE SET NULL,
    source_url TEXT,                 -- Denormalized for quick display
    content_raw TEXT,
    metadata JSONB DEFAULT '{}',     -- Type-specific: word_count, model, prompts_used
    hero_image_url TEXT,
    embedding vector(768),           -- For soft dedup
    priority_score FLOAT DEFAULT 0.0,
    published_url TEXT,
    published_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    drafted_at TIMESTAMPTZ
);

CREATE INDEX idx_artifacts_instance ON artifacts(instance_id);
CREATE INDEX idx_artifacts_status ON artifacts(status);
CREATE INDEX idx_artifacts_type ON artifacts(type);
CREATE INDEX idx_artifacts_source_ref ON artifacts(source_ref_id);
CREATE INDEX idx_artifacts_priority ON artifacts(priority_score DESC);
CREATE INDEX idx_artifacts_published_at ON artifacts(published_at DESC) WHERE published_at IS NOT NULL;
CREATE INDEX idx_artifacts_instance_status ON artifacts(instance_id, status);

CREATE TRIGGER update_artifacts_updated_at
    BEFORE UPDATE ON artifacts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
