"""LLM provider registry routing."""

from models.llm.providers.anthropic_provider import AnthropicProvider
from models.llm.providers.openrouter_provider import OpenRouterProvider
from models.llm.providers.registry import LLMProviderRegistry


def test_anthropic_claude_sonnet_resolves_to_anthropic_provider() -> None:
    provider = LLMProviderRegistry.get_provider("anthropic/claude-sonnet-4-6")
    assert isinstance(provider, AnthropicProvider)


def test_qwen_resolves_to_openrouter_provider() -> None:
    provider = LLMProviderRegistry.get_provider("qwen/qwen3.6-plus")
    assert isinstance(provider, OpenRouterProvider)
