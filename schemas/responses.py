from pydantic import BaseModel

class MoralOutrage(BaseModel):
    text_id: str
    text: str
    moral_outrage_score: int
    label_timestamp: str