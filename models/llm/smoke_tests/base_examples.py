"""Runnable example for a structured LLM classification request."""

import logging

from pydantic import BaseModel

from lib.decorators import timed
from models.llm.llm_service import LLMService

class MoralOutrage(BaseModel):
    """Structured response for a simple moral outrage classification."""

    label: int


@timed(log_level=logging.INFO)
def run_batch_example_query(texts: list[str], model: str) -> list[MoralOutrage]:
    """Run a batch of structured classification queries against gpt-5-nano."""
    llm_service = LLMService()
    prompts = [
        (
            "You are a moral outrage classifier. "
            'Return only valid JSON in the form {"label": 0} or {"label": 1}. '
            "Do not include any extra text. "
            "Set label=1 if the text expresses moral outrage, otherwise label=0. "
            f'Classify this text: "{text}"'
        )
        for text in texts
    ]
    results: list[MoralOutrage] = llm_service.structured_batch_completion(
        prompts=prompts,
        response_model=MoralOutrage,
        model=model,
    )
    return results
