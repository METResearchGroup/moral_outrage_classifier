"""Runnable example for a structured LLM classification request."""

from pydantic import BaseModel

from models.llm.llm_service import LLMService


class MoralOutrage(BaseModel):
    """Structured response for a simple moral outrage classification."""

    label: int


def run_example_query() -> MoralOutrage:
    """Run a simple structured query against gpt-5-nano."""
    llm_service = LLMService()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a moral outrage classifier. "
                "Return label=1 if the text expresses moral outrage, otherwise label=0."
            ),
        },
        {
            "role": "user",
            "content": (
                'Classify this text: "How dare they let children go hungry while '
                'executives collect bonuses."'
            ),
        },
    ]

    result = llm_service.structured_completion(
        messages=messages,
        response_model=MoralOutrage,
        model="gpt-5-nano",
    )
    print(result.model_dump_json(indent=2))
    return result


if __name__ == "__main__":
    run_example_query()
