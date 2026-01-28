# Architecture

Boundaries and contracts between components in this monorepo.

---

## Repository Structure

```
orkest/
├── backend/           # Orkest engine (generic workflow runner)
├── frontend/          # Orkest UI (generic workflow UI)
└── user_repos/
    └── dg-team/       # THIS REPO - our workflows and nodes
        ├── nodes/     # Python node functions
        ├── workflows/ # Workflow YAML definitions
        ├── shared/    # Domain models (SQLAlchemy)
        └── scripts/   # Setup, migrations
```

---

## Separation of Concerns

### Orkest Engine (`backend/`)

**What it does:**
- Executes workflows defined in YAML
- Manages workflow runs, state, routing
- Provides API for UI

**What it does NOT know:**
- Our business logic (Substack, LLM, embeddings)
- Our database schema
- Our node implementations

**Rule:** Never modify engine code for workflow-specific needs.

---

### Orkest UI (`frontend/`)

**What it does:**
- Renders workflow forms from trigger schemas
- Displays workflow runs and results
- Team/workflow selection

**What it does NOT know:**
- Workflow-specific UI components
- Our domain models
- Hardcoded workflow names

---

### dg-team (`user_repos/dg-team/`)

**What lives here:**
- Workflow YAML definitions
- Node function implementations
- Domain models (SourceItem, ProductionQueueItem, etc.)
- LLM integration, embedding logic, scrapers
- Team-specific utilities

**Rule:** All business logic lives here, not in engine.

---

## Database Boundaries

| Database | Owner | Tables |
|----------|-------|--------|
| `orkest` | Engine | `workflow_runs`, `node_executions`, `teams`, `projects`, `deployments`, `vault_configs`, `vault_secrets` |
| `dg_team` | dg-team | `source_items`, `target_items`, `production_queue`, `discovered_*`, `posts`, etc. |

**Rules:**
- Engine tables: workflow tracking, deployments, vault (configs/secrets), team/project management
- dg-team tables: domain-specific data only
- Engine never reads/writes dg-team tables
- dg-team nodes manage their own DB via `shared/models.py`
- Secrets accessed via `ctx.secrets["KEY"]` at runtime (not hardcoded)

---

## Node Contract

Nodes are the bridge between engine and business logic.

```
Engine                    dg-team
  │                          │
  │  calls function path     │
  ├─────────────────────────>│
  │                          │  nodes/db_ops.py
  │                          │  nodes/llm.py
  │                          │  nodes/scraper.py
  │  returns Pydantic model  │
  │<─────────────────────────┤
  │                          │
```

**Engine provides:**
- `ctx` object for reporting (`ctx.report_input()`, `ctx.report_output()`)
- `params` as dict (coerced to Pydantic by us)

**Nodes return:**
- Pydantic model with `status` field
- Status determines routing

---

## Adding New Components

### New Workflow
1. Create `workflows/<category>/<name>/workflow.yaml`
2. Add to `manifest.yaml`
3. Implement any new nodes in `nodes/`
4. Create Deployment in `orkest` database (see `CONVENTIONS.md` for migration example)

### New Node
1. Add function to appropriate module in `nodes/`
2. Define Input/Output schemas in `nodes/schemas.py`
3. Export from `nodes/__init__.py`

### New Database Table
1. Add SQLAlchemy model to `shared/models.py`
2. Add SQL to `schema.sql`
3. Recreate database: `dropdb dg_team && createdb dg_team && psql -d dg_team -f schema.sql`

---

## Key Files

| File | Purpose |
|------|---------|
| `manifest.yaml` | Workflow registry (what's enabled) |
| `shared/models.py` | SQLAlchemy models (our DB schema) |
| `schema.sql` | Raw SQL for DB setup (recreate DB to apply changes) |
| `nodes/schemas.py` | Pydantic I/O schemas for all nodes |
| `nodes/__init__.py` | Node function exports |
| `CONVENTIONS.md` | Developer conventions and patterns |
