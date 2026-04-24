"""Run metadata helpers."""

from typing import Any

from evaluation.model_registry import MODEL_REGISTRY
from models.llm.models import BaseHarnessLLMModel


def build_models_metadata_entries(requested_models: list[str]) -> list[dict[str, Any]]:
    """Build `metadata.json` `models` entries: llm_provider_name, resolved id, prompt fields (null for Perspective)."""
    entries: list[dict[str, Any]] = []
    for alias in requested_models:
        model_cls = MODEL_REGISTRY[alias]
        if issubclass(model_cls, BaseHarnessLLMModel):
            entries.append(
                {
                    "llm_provider_name": model_cls.get_requested_alias(),
                    "resolved_model_id": model_cls.get_resolved_model_id(),
                    "prompt_hash": model_cls.get_prompt_hash(),
                    "prompt_template": model_cls.get_prompt_template(),
                }
            )
        else:
            entries.append(
                {
                    "llm_provider_name": alias,
                    "resolved_model_id": alias,
                    "prompt_hash": None,
                    "prompt_template": None,
                }
            )
    return entries


def get_model_id_value(model_name: str) -> str:
    """Get the model ID value for the given model name.
    For LLM models, this is the resolved model ID.
    For Perspective API, this is the model name.
    """
    model_cls = MODEL_REGISTRY[model_name]
    if issubclass(model_cls, BaseHarnessLLMModel):
        return model_cls.get_resolved_model_id()
    return model_name
