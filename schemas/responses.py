from pydantic import BaseModel, Field


class MoralOutrage(BaseModel):
    text_id: str
    text: str
    moral_outrage_score: float
    label_timestamp: str


class MoralOutrageLLMResponse(BaseModel):
    """Structured LLM output for moral-outrage classification; harness maps this to MoralOutrage."""

    label: int = Field(ge=0, le=1, description="Binary moral-outrage label: 0 or 1.")