from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModelTypeVar

from models.base import BaseModel

T = TypeVar("T", bound=BaseModelTypeVar)

class BaseLLMModel(BaseModel, ABC):
    
    @abstractmethod
    def structured_completion():
        pass

    def structured_batch_completion(
        self,
        prompts: list[str],
        response_model: type[T],
        model: str | None = None,
        role: str = "user",
        **kwargs,
    ) -> list[T]:
        """
        Create batch completion requests and return Pydantic models.

        This is the main public API for structured batch completions. It orchestrates:
        1. Determining the correct provider for the model
        2. Converting prompts to message lists
        3. Executing batch completion with validation and retry logic

        Args:
            prompts: List of prompt strings
            response_model: Pydantic model class to parse each response into
            model: Model to use (default: from config, falls back to gpt-4o-mini-2024-07-18)
            role: Message role for all prompts (default: 'user')
            **kwargs: Additional parameters to pass to the API (temperature, max_tokens, etc.)
                These override any default kwargs from the model configuration.

        Returns:
            List of Pydantic model instances parsed from responses

        Raises:
            ValueError: If the model is not supported by any provider, or if any response
                content is missing or invalid (after all retries)
            ValidationError: If any response cannot be parsed into the Pydantic model
        """
        pass
