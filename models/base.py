from abc import ABC, abstractmethod
from uuid import uuid4

from schemas.responses import MoralOutrage

class BaseModel(ABC):

    @staticmethod
    def _validate_input(texts: list[str], text_ids: list[str] | None, num_rows: int) -> None:
        if not isinstance(texts, list):
            raise ValueError("Input must be a list of strings.")
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All items in the input list must be strings.")
        if text_ids is not None and len(text_ids) != num_rows:
            raise ValueError(f"text_ids length ({len(text_ids)}) must match number of texts ({num_rows}).")

    @staticmethod
    def _get_default_text_ids(num_rows) -> list[str]:
        return [str(uuid4()) for _ in range(num_rows)]

    @staticmethod
    def _get_valid_text_ids_if_not_exists(text_ids: list[str] | None, num_rows: int) -> list[str]:
        return BaseModel._get_default_text_ids(num_rows) if text_ids is None else text_ids

    @abstractmethod
    def batch_classify(
        self, 
        texts: list[str],
        text_ids: list[str] | None = None,
    ) -> list[MoralOutrage | None]:
        pass