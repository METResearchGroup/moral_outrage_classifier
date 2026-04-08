"""Runnable example for a structured LLM classification request."""

import logging

from pydantic import BaseModel

from lib.decorators import timed
from models.llm.llm_service import LLMService

example_texts = [
    "How dare they let children go hungry while executives collect bonuses.",
    "I'm so glad I'm not a parent. It's so hard to raise kids these days.",
    "Why is it always the poor who have to pay for the mistakes of the rich?",
    "It's not fair that the rich get richer and the poor get poorer.",
    "The rich are getting richer and the poor are getting poorer.",
]

class MoralOutrage(BaseModel):
    """Structured response for a simple moral outrage classification."""

    label: int


@timed(log_level=logging.INFO)
def run_batch_example_query(texts: list[str]) -> list[MoralOutrage]:
    """Run a batch of structured classification queries against gpt-5-nano."""
    llm_service = LLMService()
    prompts = [
        (
            "You are a moral outrage classifier. "
            "Return label=1 if the text expresses moral outrage, otherwise label=0. "
            f'Classify this text: "{text}"'
        )
        for text in texts
    ]
    results: list[MoralOutrage] = llm_service.structured_batch_completion(
        prompts=prompts,
        response_model=MoralOutrage,
        model="gpt-5-nano",
    )
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Batch example:")
    batch_results = run_batch_example_query(example_texts)
    for text, result in zip(example_texts, batch_results, strict=True):
        print(f"Text: {text}\tLabel: {result.label}")
        print("-" * 100)
