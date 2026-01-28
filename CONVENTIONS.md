# dg-team Developer Conventions

## For debugging go

server logs (on dev): logs/orkest.log
dg-team workflows db is DG_TEAM_DATABASE_URL user_repos/dg-team/.env
orkest db is DATABASE_URL in backend/.env

## Error Handling: Fail Loudly

**NEVER swallow errors silently.** When something fails, it should be immediately visible.

### Why?

Silent failures create debugging nightmares. When `scrape_leaderboards` returned only 2 publications instead of 165, the root cause (Pydantic validation error on `subscriber_count`) was hidden inside a generic try/except that simply skipped to the next page.

### Rules

1. **No bare `except:` clauses** - Always catch specific exceptions
2. **Log errors at the point of failure** - Include full context (what operation, what data)
3. **Re-raise or propagate errors** - Don't catch and continue unless you have a specific recovery strategy
4. **Use `ctx.report_output()` for errors** - Make failures visible in the workflow UI

### Bad Pattern

```python
try:
    pub = DiscoveredPublication(**data)
    publications.append(pub)
except Exception:
    # Silently skip - NEVER DO THIS
    continue
```

### Good Pattern

```python
try:
    pub = DiscoveredPublication(**data)
    publications.append(pub)
except ValidationError as e:
    logger.error(
        "publication_validation_failed",
        handle=data.get("handle"),
        error=str(e),
        data_sample=str(data)[:200]
    )
    # Re-raise to make the failure visible
    raise
```

### Even Better - Fix the Schema

If the API returns unexpected data formats, fix the schema to handle them properly with validators:

```python
@field_validator('subscriber_count', mode='before')
@classmethod
def parse_subscriber_count(cls, v):
    """Handle both int and string formats like 'Over 80,000 subscribers'"""
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        # Parse string format
        ...
```

---

## Verbose Reporting

All node functions should provide detailed `ctx.report_input()` and `ctx.report_output()` calls.

### Why?

- Makes debugging easier - can see exactly what data flowed through each node
- Gives visibility into workflow progress
- Helps identify where things went wrong

### Example

```python
async def some_node(ctx, params):
    ctx.report_input({
        "target_handle": params.target_handle,
        "items_count": len(params.items),
        "first_item": params.items[0] if params.items else None,
    })

    # ... do work ...

    ctx.report_output({
        "processed_count": len(results),
        "failed_count": failed,
        "sample_result": results[0] if results else None,
    })
```

---

## API Response Handling

When parsing API responses, always validate the expected structure.

### Substack API Specifics

- Leaderboard API structure: `data.items[].publication` (always nested)
- The `publication` field contains the actual publication data
- Never assume flat structure like `data.items[]` directly having publication fields

### Example

```python
# CORRECT - Substack API nests publication data
for item in data.get("items", []):
    pub_data = item.get("publication", {})
    if pub_data:
        handle = pub_data.get("subdomain")

# WRONG - Assuming flat structure
for item in data.get("items", []):
    handle = item.get("subdomain")  # This won't work!
```

---

## Data Quality: No Empty Data

**NEVER store or process empty/incomplete data.** If we can't get proper data, fail rather than continue with garbage.

### Why?

Empty data pollutes downstream processing. When posts without content get stored and embedded, they create meaningless vectors that match random queries. A title-only embedding for "Live with Eric Schmidt from Davos" will falsely match "System Design" queries because short text creates shallow, generic embeddings.

### Rules

1. **Validate content before storing** - Posts must have minimum content length (e.g., ≥100 chars)
2. **Skip incomplete records** - Don't store posts without content just to have a row in the DB
3. **Log skipped items** - Make it visible what was skipped and why
4. **Don't cache failures** - If content extraction fails, don't cache the empty result

### Bad Pattern

```python
# Store post even if content is empty - NEVER DO THIS
content_raw = post.content_raw or ""
db_post = DBPost(
    title=post.title,
    content_raw=content_raw,  # Could be empty!
    embedding=embed(title + content_raw)  # Garbage embedding
)
session.add(db_post)
```

### Good Pattern

```python
# STRICT: Skip posts without meaningful content
content_raw = post.content_raw or ""
if len(content_raw) < 100:
    logger.warning("skipping_post_no_content", url=post.url, content_len=len(content_raw))
    skipped_posts += 1
    continue

# Only store posts with real content
db_post = DBPost(
    title=post.title,
    content_raw=content_raw,
    embedding=embed(title + content_raw)
)
session.add(db_post)
```

---

## Caching Strategy

- **DO cache:** Article content (expensive to fetch, rarely changes)
- **DON'T cache:** Leaderboard/discovery APIs (data changes frequently, need fresh results)
- **DON'T cache failures:** If fetching fails, don't cache the empty/error response

---

## Pydantic Schemas for Node Functions

**All node functions MUST use Pydantic schemas for input and output types.**

### Why?

- **Type safety**: Catches type errors early via validation
- **Documentation**: Schemas serve as documentation for node interfaces
- **Consistency**: All workflows use the same pattern
- **IDE support**: Better autocomplete and type checking

### Rules

1. **Define Input/Output schemas** in `nodes/schemas.py` for every node
2. **Use type hints** on the function signature: `params: MyInput` and `-> MyOutput`
3. **No manual dict handling** - the engine coerces dicts to Pydantic models automatically
4. **Export schemas** from `nodes/schemas.py` and import in the node file

### Good Pattern

```python
# In schemas.py
class FetchTopSourcePostsInput(BaseModel):
    """Input for fetch_top_source_posts node."""
    target_handle: str = Field(default="", description="Target publication handle")
    posts_per_source: int = Field(default=5, description="Posts to fetch per source")
    max_age_days: int = Field(default=30, description="Only posts within this timeframe")

class FetchTopSourcePostsOutput(BaseModel):
    """Output from fetch_top_source_posts node."""
    source_posts: List[SourcePost] = Field(default_factory=list)
    total_count: int = 0
    sources_count: int = 0
    status: str = "success"

# In db_ops.py
async def fetch_top_source_posts(
    ctx,
    params: FetchTopSourcePostsInput,
) -> FetchTopSourcePostsOutput:
    """Fetch top-performing posts from source publications."""
    target_handle = params.target_handle  # Direct attribute access
    posts_per_source = params.posts_per_source
    max_age_days = params.max_age_days
    # ... implementation ...
```

### Bad Pattern (Legacy - Do Not Use)

```python
# DON'T do this - manual dict handling
async def fetch_top_source_posts(ctx, params):
    if isinstance(params, dict):
        target_handle = params.get("target_handle", "")
    else:
        target_handle = params.target_handle
```

---

## Workflow Registration

**All workflows must be registered in `manifest.yaml` to be discoverable by the engine.**

### Why?

The Orkest engine doesn't auto-discover workflows by scanning directories. It reads `manifest.yaml` to get the list of enabled workflows. If you create a new workflow directory but don't add it to the manifest, it won't appear in the UI.

### manifest.yaml Structure

```yaml
name: "dg-team"
version: "2.0"

workflows:
  - path: "workflows/substack/content_sync"
    enabled: true
    description: "Sync state of source and target publications"

  - path: "workflows/substack/my_new_workflow"
    enabled: true
    description: "What this workflow does"
```

### Checklist for New Workflows

1. Create directory: `workflows/<category>/<workflow_name>/`
2. Create `workflow.yaml` inside that directory
3. **Add entry to `manifest.yaml`** with:
   - `path`: Relative path to workflow directory (not the yaml file)
   - `enabled`: Set to `true`
   - `description`: Brief description
4. Restart backend to pick up changes

### Disabling Workflows

Set `enabled: false` in manifest.yaml to hide a workflow from the UI without deleting it:

```yaml
- path: "workflows/substack/old_workflow"
  enabled: false # Hidden but not deleted
  description: "Deprecated workflow"
```

---

## Deployments & Vault Configuration

**Workflows need Deployments to be runnable from the UI.** A workflow.yaml defines *what* can happen; a Deployment defines *how* it runs (with what configuration).

### Why Deployments?

- **Same workflow, different configs**: Run `factory` workflow for both `postsyntax` and `theneuralmaze` projects with different settings
- **Input defaults**: Pre-fill workflow inputs so users don't have to enter them every time
- **Secrets binding**: Securely provide API keys without hardcoding them

### Creating a Deployment

Deployments are stored in the `orkest` database (not the dg-team database). Create via UI: Deployments page → Create Deployment.

### Input Resolution Priority

When a deployment runs, inputs are merged in this order (highest priority wins):

```
YAML defaults < input_config < vault_bindings < run_params
```

1. **YAML trigger schema defaults**: Defined in workflow.yaml `trigger.schema.properties.*.default`
2. **input_config**: JSON from the vault config specified by `input_config` slug
3. **vault_bindings**: Individual config values bound to specific inputs
4. **run_params**: Values provided at trigger time (manual overrides)

### Checklist for New Workflows

1. ✅ Create `workflows/<category>/<name>/workflow.yaml`
2. ✅ Add to `manifest.yaml`
3. ✅ Create vault config with sensible defaults (via UI)
4. ✅ Create deployment binding workflow to config (via UI)

### Database Locations

| Data | Database | Connection |
|------|----------|------------|
| Deployments, Vault configs/secrets, Projects | `orkest` | `DATABASE_URL` in `backend/.env` |
| Posts, Publications, dg-team business data | `dg-team` | `DG_TEAM_DATABASE_URL` in `user_repos/dg-team/.env` |

---

## Workflow Data Architecture

### Philosophy: Context vs External Resources

A workflow run has two categories of data:

**1. Context (Workflow State)**
- The "blackboard" - data flowing between nodes
- Includes: trigger inputs, node outputs, intermediate state
- Stored on `workflow_run.input_params` and `workflow_run.final_context`
- Passed to each node execution
- **This is the only data that travels with the workflow**

**2. External Resources (Accessed via API)**
- Configs, secrets, KV store
- NOT part of context - fetched on-demand via API calls
- Accessed via `ctx.get_config()`, `ctx.get_secret()`, `ctx.kv.get()`

### Why This Separation Matters

**Future architecture:** User code will run in isolated containers. Each node may execute on a different machine. The only thing passed between nodes is context. External resources are fetched via API.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Machine A     │     │   Machine B     │     │   Machine C     │
│   Node 1        │     │   Node 2        │     │   Node 3        │
│                 │     │                 │     │                 │
│  ctx.state ─────┼────►│  ctx.state ─────┼────►│  ctx.state      │
│  (passed)       │     │  (passed)       │     │  (passed)       │
│                 │     │                 │     │                 │
│  ctx.get_config │     │  ctx.get_config │     │  ctx.get_config │
│      ↓          │     │      ↓          │     │      ↓          │
│  [API call]     │     │  [API call]     │     │  [API call]     │
└────────┼────────┘     └────────┼────────┘     └────────┼────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ↓
                    ┌────────────────────────┐
                    │     Orkest Backend     │
                    │  ┌──────┐ ┌──────────┐ │
                    │  │Vault │ │ KV Store │ │
                    │  └──────┘ └──────────┘ │
                    └────────────────────────┘
```

### Data Mechanisms

| Mechanism | Use For | Storage | Access Pattern |
|-----------|---------|---------|----------------|
| `input_params` | Runtime values (target_handle, model, flags) | workflow_run.input_params | Passed in context |
| `input_config` | Pre-filling form defaults | Resolved → input_params | Passed in context |
| `ctx.get_config()` | Large prompts, templates, static config | Vault configs | Lazy-load via API |
| `ctx.get_secret()` | API keys, credentials | Vault secrets | Lazy-load via API |
| `ctx.kv` | Persistent state across runs | KV store | API calls |

### Configs: Lazy-Loading Design

Configs are **not** resolved at workflow start and passed in memory. Instead:

1. **Deployment** defines config bindings: `{"article_prompt": "article-prompt-slug"}`
2. **Workflow run** stores the mapping (not values): `workflow_run.config_bindings`
3. **At runtime**, when node calls `ctx.get_config("article_prompt")`:
   - SDK looks up slug from mapping
   - Calls Orkest API to fetch value from vault
   - Caches result for that node execution

**Why lazy-load?**
- Keeps context lean (no 10KB prompts in workflow state)
- Works in distributed execution (each machine fetches what it needs)
- Retries work automatically (mapping stored on run, values fetched fresh)
- Config updates in vault reflect on next node execution

### Example: Workflow with Configs

**workflow.yaml:**
```yaml
configs:
  article_prompt:
    description: "Writing style instructions"
    required: true
    type: PROMPT_TEMPLATE
  hero_image_style:
    description: "Hero image generation style"
    required: true
    type: PROMPT_TEMPLATE
```

**Deployment binding (UI):**
```json
{
  "vault_bindings": {
    "configs": {
      "article_prompt": "article-prompt",
      "hero_image_style": "hero-image-prompt"
    }
  }
}
```

**Node code:**
```python
async def draft_article(ctx, params: DraftArticleInput) -> DraftArticleOutput:
    # Lazy-loads from vault via API
    prompt_config = ctx.get_config("article_prompt")
    prompt_text = prompt_config.get("text", "")

    # Use prompt...
```

### What NOT to Do

❌ **Don't pass large prompts as input_params**
```yaml
# Bad - 2000 word prompt in params
trigger:
  schema:
    properties:
      article_prompt:
        type: string
```

❌ **Don't store config values in context**
```python
# Bad - pollutes workflow state
ctx.state["cached_prompt"] = ctx.get_config("article_prompt")
```

❌ **Don't read vault directly from node code**
```python
# Bad - bypasses abstraction, breaks in distributed mode
from shared.db import get_db
config = await db.query(VaultConfig).filter_by(slug="...").first()
```

✅ **Do use ctx.get_config() for all config access**
```python
# Good - works locally and distributed
prompt = ctx.get_config("article_prompt")
```    