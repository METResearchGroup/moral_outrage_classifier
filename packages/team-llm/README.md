# team-llm

Internal Python package for structured LLM completions via [LiteLLM](https://github.com/BerriAI/litellm), with provider routing, YAML model configuration, retries, and Pydantic response validation.

## Quick start

```python
from pydantic import BaseModel
from team_llm import LLMService

class LabelResponse(BaseModel):
    label: int

service = LLMService()
result = service.structured_completion(
    messages=[{"role": "user", "content": "Classify this text."}],
    response_model=LabelResponse,
    model="gpt-5.4-nano",
)
print(result.label)
```

Set provider API keys in your environment (for example `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`). Bedrock models use AWS credentials instead of an API key.

## Documentation

See [SETUP.md](./SETUP.md) for:

- Creating a standalone GitHub repository from this package
- Local development setup with `uv`
- Installing the package in other projects (editable, Git, and wheel)
