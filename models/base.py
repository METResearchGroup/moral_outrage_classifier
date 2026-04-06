from abc import ABC, abstractmethod

from schemas.responses import MoralOutrage

class BaseModel(ABC):

    @staticmethod
    def _validate_input(texts: list[str]) -> None:
        if not isinstance(texts, list):
            raise ValueError("Input must be a list of strings.")
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All items in the input list must be strings.")

    @abstractmethod
    def batch_classify(self, texts: list[str]) -> list[MoralOutrage | None]:
        pass