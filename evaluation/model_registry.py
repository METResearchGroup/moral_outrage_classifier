"""Registry of evaluation model aliases to classifier classes."""

from models.base import BaseModel as ClassifierBaseModel
from models.llm.models import (
    AnthropicModel,
    MiniMaxModel,
    OpenAIModel,
    QwenModel,
)
from models.perspective_api.model import PerspectiveAPIModel

MODEL_REGISTRY: dict[str, type[ClassifierBaseModel]] = {
    "perspective_api": PerspectiveAPIModel,
    "openai": OpenAIModel,
    "qwen": QwenModel,
    "anthropic": AnthropicModel,
    "minimax": MiniMaxModel,
}

VALID_MODELS = list(MODEL_REGISTRY.keys())
