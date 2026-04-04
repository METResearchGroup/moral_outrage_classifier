from abc import ABC, abstractmethod

from schemas.responses import MoralOutrage

class BaseModel(ABC):

    @abstractmethod
    def batch_classify(self, texts: list[str]) -> list[MoralOutrage | None]:
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise ValueError("Input must be a list of strings.")