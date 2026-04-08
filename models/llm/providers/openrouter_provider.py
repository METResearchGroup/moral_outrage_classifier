"""OpenRouter provider implementation."""

from typing import Any

from pydantic import BaseModel

from lib.load_env_vars import EnvVarsContainer
from models.llm.providers.base import LLMProviderProtocol


class OpenRouterProvider(LLMProviderProtocol):
    """OpenRouter provider implementation.

    Handles OpenRouter-specific logic:
    - API key management
    """

    def __init__(self):
        self._initialized = False
        self._api_key: str | None = None

    @property
    def provider_name(self) -> str:
        return "openrouter"

    @property
    def supported_models(self) -> list[str]:
        return []

    @property
    def api_key(self) -> str:
        if self._api_key is None:
            raise RuntimeError(
                "OpenAIProvider has not been initialized with an API key. "
                "Call initialize() before making LiteLLM requests."
            )
        return self._api_key

    def initialize(self, api_key: str | None = None) -> None:
        if api_key is None:
            api_key = EnvVarsContainer.get_env_var("OPENAI_API_KEY", required=True)
        if not self._initialized:
            self._api_key = api_key
            self._initialized = True

    def supports_model(self, model_name: str) -> bool:
        return model_name in self.supported_models

    def format_structured_output(
        self,
        response_model: type[BaseModel],
        model_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Format OpenAI's structured output format.

        OpenAI requires:
        - type: "json_schema"
        - strict: True
        - schema with additionalProperties: false on all objects
        """
        schema = response_model.model_json_schema()
        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__.lower(),
                "strict": True,
                "schema": schema,
            },
        }

    def prepare_completion_kwargs(
        self,
        model: str,
        messages: list[dict],
        response_format: dict[str, Any] | None,
        model_config: dict[str, Any],
        **kwargs,
    ) -> dict[str, Any]:
        """Prepare completion kwargs."""
        if not self._initialized:
            self.initialize()

        # Merge model_config defaults with user kwargs (user kwargs take precedence)
        merged_kwargs = {**model_config.get("kwargs", {}), **kwargs}

        completion_kwargs = {
            "model": model,
            "messages": messages,
            **merged_kwargs,
        }

        if response_format is not None:
            completion_kwargs["response_format"] = response_format

        return completion_kwargs
