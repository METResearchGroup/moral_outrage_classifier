"""Harness-facing LLM facades."""

import hashlib
from typing import ClassVar

from lib.timestamp_utils import get_current_timestamp
from models.base import BaseModel as ClassifierBaseModel
from models.llm.llm_service import LLMService, get_llm_service
from models.llm.prompts import MORAL_OUTRAGE_PROMPT
from schemas.responses import MoralOutrage, MoralOutrageLLMResponse


class BaseHarnessLLMModel(ClassifierBaseModel):
    """Shared facade for all LLM models."""

    alias: ClassVar[str] # e.g. "openai", "qwen", "anthropic", "minimax"
    resolved_model_id: ClassVar[str] # e.g. "gpt-5.4", "qwen/qwen3.6-plus", "anthropic/claude-sonnet-4-6", "minimax/minimax-m2.5"

    def __init__(self, llm_service: LLMService | None = None) -> None:
        self._llm_service = llm_service or get_llm_service()

    @property
    def llm_service(self) -> LLMService:
        return self._llm_service

    @classmethod
    def get_requested_alias(cls) -> str:
        return cls.alias

    @classmethod
    def get_resolved_model_id(cls) -> str:
        return cls.resolved_model_id

    @classmethod
    def get_prompt_template(cls) -> str:
        return MORAL_OUTRAGE_PROMPT

    @classmethod
    def get_prompt_hash(cls) -> str:
        """Hash the prompt template to ensure consistency.
        
        We get the hash of the raw prompt template.
        """
        raw = cls.get_prompt_template()
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def render_prompt(self, text: str) -> str:
        return self.get_prompt_template().replace("{{input}}", text)

    def batch_classify(
        self,
        texts: list[str],
        text_ids: list[str] | None = None,
    ) -> list[MoralOutrage | None]:
        self._validate_input(texts, text_ids, len(texts))
        text_ids: list[str] = self._get_valid_text_ids_if_not_exists(text_ids, len(texts))
        prompts: list[str] = [self.render_prompt(text) for text in texts]
        responses: list[MoralOutrageLLMResponse] = self.llm_service.structured_batch_completion(
            prompts=prompts,
            response_model=MoralOutrageLLMResponse,
            model=self.get_resolved_model_id(),
        )
        timestamp: str = get_current_timestamp()
        return [
            MoralOutrage(
                text_id=tid,
                text=text,
                moral_outrage_score=float(resp.label),
                label_timestamp=timestamp,
            )
            for text, tid, resp in zip(texts, text_ids, responses, strict=True)
        ]


class OpenAIModel(BaseHarnessLLMModel):
    alias: ClassVar[str] = "openai"
    resolved_model_id: ClassVar[str] = "gpt-5.4"


class QwenModel(BaseHarnessLLMModel):
    alias: ClassVar[str] = "qwen"
    resolved_model_id: ClassVar[str] = "qwen/qwen3.6-plus"


class AnthropicModel(BaseHarnessLLMModel):
    alias: ClassVar[str] = "anthropic"
    resolved_model_id: ClassVar[str] = "anthropic/claude-sonnet-4-6"


class MiniMaxModel(BaseHarnessLLMModel):
    alias: ClassVar[str] = "minimax"
    resolved_model_id: ClassVar[str] = "minimax/minimax-m2.5"
