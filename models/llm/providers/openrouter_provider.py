"""OpenRouter provider implementation."""

import copy
from typing import Any

from pydantic import BaseModel

from lib.load_env_vars import EnvVarsContainer
from models.llm.config.model_registry import ModelConfigRegistry
from models.llm.providers.base import LLMProviderProtocol


class OpenRouterProvider(LLMProviderProtocol):
    """OpenRouter provider implementation.

    Handles OpenRouter-specific logic:
    - API key management (OPENROUTER_API_KEY)
    - Strict structured-output schema patching (same rules as OpenAI)
    - LiteLLM model IDs: openrouter/<vendor>/<model>
    """

    def __init__(self):
        self._initialized = False
        self._api_key: str | None = None

    @property
    def provider_name(self) -> str:
        return "openrouter"

    @property
    def supported_models(self) -> list[str]:
        return ModelConfigRegistry.list_models_for_provider("openrouter")

    @property
    def api_key(self) -> str:
        if self._api_key is None:
            raise RuntimeError(
                "OpenRouterProvider has not been initialized with an API key. "
                "Call initialize() before making LiteLLM requests."
            )
        return self._api_key

    def initialize(self, api_key: str | None = None) -> None:
        if api_key is None:
            api_key = EnvVarsContainer.get_env_var("OPENROUTER_API_KEY", required=True)
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
        """Format OpenAI-compatible structured output for OpenRouter.

        Strict json_schema requires additionalProperties: false on all objects.
        """
        schema = response_model.model_json_schema()
        fixed_schema = self._fix_schema_for_strict_mode(schema)

        return {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__.lower(),
                "strict": True,
                "schema": fixed_schema,
            },
        }

    def _supports_native_structured_output(self, model: str) -> bool:
        """Return whether the model reliably supports native json_schema mode."""
        return model in {
            "anthropic/claude-sonnet-4.6",
        }

    def _litellm_model_id(self, public_model_id: str) -> str:
        """Map public config ID (e.g. qwen/qwen3.6-plus) to LiteLLM OpenRouter form."""
        if public_model_id.startswith("openrouter/"):
            return public_model_id
        return f"openrouter/{public_model_id}"

    def prepare_completion_kwargs(
        self,
        model: str,
        messages: list[dict],
        response_format: dict[str, Any] | None,
        model_config: dict[str, Any],
        **kwargs,
    ) -> dict[str, Any]:
        """Prepare completion kwargs for LiteLLM OpenRouter routing."""
        if not self._initialized:
            self.initialize()

        # Merge model_config defaults with user kwargs (user kwargs take precedence)
        merged_kwargs = {**model_config.get("kwargs", {}), **kwargs}

        completion_kwargs = {
            "model": self._litellm_model_id(model),
            "messages": messages,
            **merged_kwargs,
        }

        if response_format is not None and self._supports_native_structured_output(
            model
        ):
            completion_kwargs["response_format"] = response_format

        return completion_kwargs

    def _fix_schema_for_strict_mode(self, schema: dict) -> dict:
        """Recursively add additionalProperties: false to all object definitions."""
        schema_copy = copy.deepcopy(schema)
        self._patch_recursive(schema_copy)
        return schema_copy

    def _patch_recursive(self, obj) -> None:
        """Recursively patch schema, handling dicts and lists."""
        if isinstance(obj, dict):
            if obj.get("type") == "object":
                obj["additionalProperties"] = False
            for value in obj.values():
                self._patch_recursive(value)
        elif isinstance(obj, list):
            for item in obj:
                self._patch_recursive(item)
