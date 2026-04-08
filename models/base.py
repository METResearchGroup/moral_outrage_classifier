from abc import ABC, abstractmethod
from uuid import uuid4

from schemas.responses import MoralOutrage

class BaseModel(ABC):

    @staticmethod
    def _validate_input(texts: list[str]) -> None:
        if not isinstance(texts, list):
            raise ValueError("Input must be a list of strings.")
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All items in the input list must be strings.")

    @staticmethod
    def _validate_text_ids(text_ids: list[str] | None, num_rows) -> list[str]:
        if text_ids is not None:
            return text_ids
        return [str(uuid4()) for i in range(num_rows)]

    @abstractmethod
    def batch_classify(
        self, 
        texts: list[str],
        text_ids: list[str] | None = None,
    ) -> list[MoralOutrage | None]:
        pass