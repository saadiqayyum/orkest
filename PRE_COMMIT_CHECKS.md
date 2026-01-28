# Pre-Commit Checks

Run these checks before committing changes to the repository.

## 1. Dead Code

- [ ] No unused imports (check Pylance warnings in VS Code)
- [ ] No unused variables or assignments
- [ ] No orphaned functions that are never called
- [ ] No commented-out code blocks

## 2. Pydantic Schemas

  - [ ] All node functions use Pydantic Input/Output schemas (see `CONVENTIONS.md`)
  - [ ] No manual `isinstance(params, dict)` checks
- [ ] Type hints use actual classes, not forward reference strings
- [ ] Pydantic objects converted to dicts via `model_dump()` before returning to workflow context

## 3. Error Handling

- [ ] No bare `except:` clauses
- [ ] Errors logged with context before re-raising
- [ ] No silent failures (errors swallowed in try/except)

## 4. Context Reporting

- [ ] All node functions call `ctx.report_input()` with relevant data
- [ ] All node functions call `ctx.report_output()` with results
- [ ] Report data is comprehensive (not just counts, but sample items)

## 5. LLM Calls

- [ ] JSON responses validated via `call_llm_validated()` with Pydantic model
- [ ] Prompts use XML-style tags for data boundaries (`<source_context>`, etc.)
- [ ] Response schemas generated dynamically from Pydantic: `model_json_schema()`
- [ ] Token usage minimized (only essential fields in context)

## 6. Workflow Registration

- [ ] New workflows added to `manifest.yaml`
- [ ] Workflow paths are relative directories (not yaml files)

## 7. Git Hygiene

- [ ] No `.env` or credentials files staged
- [ ] No large binary files
- [ ] Commit message describes "why", not just "what"

## Quick Commands

```bash
# Check for Pylance/type errors
# (Use VS Code or run mypy if configured)

# Check git status
git status

# Check what's being committed
git diff --cached
```

---

## Dev Tools

### LLM Dev Cache

Speed up development by caching LLM responses. All calls through `call_llm_validated()` are automatically cached based on prompt + model.

```bash
# In .env
LLM_DEV_CACHE=true
```

Cache files stored in `/tmp/llm_dev_cache/llm_<hash>.json`

**Clear cache:**
```bash
# Clear all LLM cache
rm -rf /tmp/llm_dev_cache/
```

Cache is keyed on `hash(prompt + model)` - same prompt = cache hit.
