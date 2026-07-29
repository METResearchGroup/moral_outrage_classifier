"""Environment variable registration for LLM providers."""

from team_llm.env import ENV_VAR_TYPES


def test_anthropic_api_key_registered() -> None:
    assert "ANTHROPIC_API_KEY" in ENV_VAR_TYPES
    assert ENV_VAR_TYPES["ANTHROPIC_API_KEY"] is str
