# team-llm setup and installation

This document explains how to publish `team-llm` as its own GitHub repository, develop it locally with `uv`, and install it from other projects.

## What is in this package?

`team-llm` is a small internal library that wraps [LiteLLM](https://github.com/BerriAI/litellm) for structured completions:

- `LLMService` for single and batch structured completions validated with Pydantic
- Provider routing for OpenAI, Anthropic, OpenRouter, and Amazon Bedrock
- YAML-backed model configuration (`src/team_llm/config/models.yaml`)
- Retry logic with provider-aware rate-limit handling
- Internal exception types for consistent error handling

Import path:

```python
from team_llm import LLMService, get_llm_service
```

## 1. Create a standalone GitHub repository

1. Create a new empty repository in your GitHub organization, for example `your-org/team-llm`.
2. Copy the contents of `packages/team-llm/` from this monorepo into the new repository root. The new repo should look like:

```text
team-llm/
├── pyproject.toml
├── README.md
├── SETUP.md
├── src/
│   └── team_llm/
│       ├── __init__.py
│       ├── llm_service.py
│       ├── exceptions.py
│       ├── retry.py
│       ├── env.py
│       ├── aws/
│       ├── config/
│       └── providers/
└── tests/
```

3. Commit and push:

```bash
git init
git add .
git commit -m "Initial import of team-llm package"
git branch -M main
git remote add origin git@github.com:your-org/team-llm.git
git push -u origin main
```

4. Tag releases when you want consumers to pin versions:

```bash
git tag v0.1.0
git push origin v0.1.0
```

## 2. Local development setup (uv)

From the package root:

```bash
cd team-llm
uv sync --extra test
```

Run the package tests:

```bash
uv run pytest
```

Build a wheel locally:

```bash
uv build
```

The built artifacts appear in `dist/`.

### Environment variables

Providers read credentials from the environment (or a `.env` file loaded by `python-dotenv`):

| Variable | Used by |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI models |
| `ANTHROPIC_API_KEY` | Anthropic models |
| `OPENROUTER_API_KEY` | OpenRouter models |
| `AWS_PROFILE` / default AWS credential chain | Bedrock models |

## 3. Install in another project

All examples below assume the consuming project also uses `uv` and `pyproject.toml`.

### Option A: Editable install from a local path (monorepo or sibling checkout)

In the consuming project's `pyproject.toml`:

```toml
[project]
dependencies = [
    "team-llm",
]

[tool.uv.sources]
team-llm = { path = "../team-llm", editable = true }
```

Then sync:

```bash
uv sync
```

This is how `moral-outrage-classifier` consumes the package while it still lives under `packages/team-llm/`.

### Option B: Install directly from GitHub

Pin to a branch:

```toml
[project]
dependencies = [
    "team-llm",
]

[tool.uv.sources]
team-llm = { git = "https://github.com/your-org/team-llm.git", branch = "main" }
```

Or pin to a release tag:

```toml
[tool.uv.sources]
team-llm = { git = "https://github.com/your-org/team-llm.git", tag = "v0.1.0" }
```

Then:

```bash
uv sync
```

### Option C: Install from a built wheel

After `uv build`, publish the wheel to your internal package index or attach it to a GitHub Release. Consumers can install with:

```bash
uv add ./dist/team_llm-0.1.0-py3-none-any.whl
```

or reference the hosted wheel URL in `[tool.uv.sources]`.

## 4. Basic usage after installation

```python
from pydantic import BaseModel
from team_llm import LLMService

class LabelResponse(BaseModel):
    label: int

service = LLMService()
result = service.structured_completion(
    messages=[{"role": "user", "content": "Return a label for this text."}],
    response_model=LabelResponse,
    model="gpt-5.4-nano",
)
print(result.label)
```

Batch API:

```python
results = service.structured_batch_completion(
    prompts=["first prompt", "second prompt"],
    response_model=LabelResponse,
    model="gpt-5.4-nano",
)
```

## 5. Customizing model configuration

Default model settings live in `src/team_llm/config/models.yaml`. To override the config path at runtime (for tests or deployment-specific configs):

```python
from pathlib import Path
from team_llm.config.model_registry import ModelConfigRegistry

ModelConfigRegistry.set_config_path(Path("/path/to/custom-models.yaml"))
```

## 6. Backward compatibility in this monorepo

While `team-llm` is vendored under `packages/team-llm/`, the original import paths under `models.llm` remain as thin re-exports. New code should import from `team_llm` directly.

## 7. Releasing a new version

1. Bump `version` in `pyproject.toml`.
2. Run `uv run pytest`.
3. Run `uv build`.
4. Commit, tag (`v0.1.1`), and push the tag.
5. Update consuming projects to the new tag or republish the wheel.
