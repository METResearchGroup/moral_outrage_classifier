"""Amazon Bedrock provider implementation."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel

from lib.aws.bedrock import BedrockClient
from models.llm.config.model_registry import ModelConfigRegistry
from models.llm.providers.base import LLMProviderProtocol


class BedrockProvider(LLMProviderProtocol):
    """Amazon Bedrock provider for LiteLLM-routed foundation models."""

    MODEL_ID_MAP = {
        "qwen/qwen3.6-plus": "bedrock/qwen.qwen3-32b-v1:0",
    }

    def __init__(self) -> None:
        self._initialized = False
        self.client: BedrockClient | None = None

    @property
    def provider_name(self) -> str:
        return "bedrock"

    @property
    def supported_models(self) -> list[str]:
        return ModelConfigRegistry.list_models_for_provider("bedrock")

    @property
    def api_key(self) -> str | None:
        return None

    def initialize(self, api_key: str | None = None) -> None:
        if not self._initialized:
            self.client = BedrockClient()
            self._initialized = True

    def supports_model(self, model_name: str) -> bool:
        return model_name in self.supported_models

    def format_structured_output(
        self,
        response_model: type[BaseModel],
        model_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Format OpenAI-compatible strict JSON schema for LiteLLM."""
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
        """Prepare completion kwargs for LiteLLM Bedrock routing."""
        if not self._initialized:
            self.initialize()

        merged_kwargs = {**model_config.get("kwargs", {}), **kwargs}

        completion_kwargs: dict[str, Any] = {
            "model": self._litellm_model_id(model),
            "messages": messages,
            **merged_kwargs,
        }

        if response_format is not None:
            completion_kwargs["response_format"] = response_format

        return completion_kwargs

    def _litellm_model_id(self, public_model_id: str) -> str:
        try:
            return self.MODEL_ID_MAP[public_model_id]
        except KeyError as exc:
            raise ValueError(
                f"No Bedrock LiteLLM model mapping configured for {public_model_id!r}."
            ) from exc

    def _fix_schema_for_strict_mode(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Recursively add additionalProperties: false to all object definitions."""
        schema_copy = copy.deepcopy(schema)
        self._patch_recursive(schema_copy)
        return schema_copy

    def _patch_recursive(self, obj: Any) -> None:
        if isinstance(obj, dict):
            if obj.get("type") == "object":
                obj["additionalProperties"] = False
            for value in obj.values():
                self._patch_recursive(value)
        elif isinstance(obj, list):
            for item in obj:
                self._patch_recursive(item)
