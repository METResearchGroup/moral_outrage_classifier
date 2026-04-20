"""Anthropic provider implementation."""

import copy
from typing import Any

from pydantic import BaseModel

from lib.load_env_vars import EnvVarsContainer
from models.llm.config.model_registry import ModelConfigRegistry
from models.llm.providers.base import LLMProviderProtocol


class AnthropicProvider(LLMProviderProtocol):
    """Anthropic provider: API key from ANTHROPIC_API_KEY, models from config."""

    def __init__(self) -> None:
        self._initialized = False
        self._api_key: str | None = None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def supported_models(self) -> list[str]:
        return ModelConfigRegistry.list_models_for_provider("anthropic")

    @property
    def api_key(self) -> str:
        if self._api_key is None:
            raise RuntimeError(
                "AnthropicProvider has not been initialized with an API key. "
                "Call initialize() before making LiteLLM requests."
            )
        return self._api_key

    def initialize(self, api_key: str | None = None) -> None:
        if api_key is None:
            api_key = EnvVarsContainer.get_env_var("ANTHROPIC_API_KEY", required=True)
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
        """Format strict json_schema for Anthropic (additionalProperties: false on objects)."""
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

    def prepare_completion_kwargs(
        self,
        model: str,
        messages: list[dict],
        response_format: dict[str, Any] | None,
        model_config: dict[str, Any],
        **kwargs,
    ) -> dict[str, Any]:
        if not self._initialized:
            self.initialize()

        merged_kwargs = {**model_config.get("kwargs", {}), **kwargs}

        completion_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            **merged_kwargs,
        }

        if response_format is not None:
            completion_kwargs["response_format"] = response_format

        return completion_kwargs

    def _fix_schema_for_strict_mode(self, schema: dict) -> dict:
        """Recursively add additionalProperties: false to all object definitions."""
        schema_copy = copy.deepcopy(schema)
        self._patch_recursive(schema_copy)
        return schema_copy

    def _patch_recursive(self, obj) -> None:
        if isinstance(obj, dict):
            if obj.get("type") == "object":
                obj["additionalProperties"] = False
            for value in obj.values():
                self._patch_recursive(value)
        elif isinstance(obj, list):
            for item in obj:
                self._patch_recursive(item)
