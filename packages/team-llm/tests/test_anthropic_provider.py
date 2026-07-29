"""AnthropicProvider behavior."""

import pytest

from team_llm.env import EnvVarsContainer
from team_llm.providers.anthropic_provider import AnthropicProvider


def test_prepare_completion_kwargs_forwards_response_format_when_present() -> None:
    provider = AnthropicProvider()
    provider.initialize(api_key="test-key")
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "dummyresponse",
            "strict": True,
            "schema": {"type": "object", "properties": {"label": {"type": "integer"}}},
        },
    }
    out = provider.prepare_completion_kwargs(
        model="anthropic/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "x"}],
        response_format=response_format,
        model_config={"kwargs": {}},
    )
    assert out["response_format"] == response_format


def test_initialize_without_api_key_requires_anthropic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_env_var(cls, name: str, required: bool = False) -> str:
        if name == "ANTHROPIC_API_KEY" and required:
            raise ValueError(
                "ANTHROPIC_API_KEY is required but is missing. "
                "Please set the ANTHROPIC_API_KEY environment variable."
            )
        return ""

    monkeypatch.setattr(
        EnvVarsContainer,
        "get_env_var",
        classmethod(fake_get_env_var),
    )
    provider = AnthropicProvider()
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        provider.initialize()
